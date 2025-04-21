from utils import *
from esm.modeling_esm_geoupdate import EsmModel
from geo import ExpNormalSmearing
from einops import repeat

class SI_tuning(nn.Module):
    def __init__(
        self,
        num_labels,
        loss_fn,
        config_path,
        model_size='650',
        task='classification',
        fusion_dim=16,
        plddt=False,
        b_angle=True,
        b_coord=True,
        lora_r=16,
        peft=True
    ):
        """
        Args:
            num_labels: Number of output labels
            loss_fn: Loss function ('BCE', 'MSE', or 'CE')
            config_path: Path to pretrained model config
            model_size: Model size ('650', '35', or 'saprot')
            task: Task type ('classification' or 'regression')
            fusion_dim: Dimension for fusion layer
            plddt: Whether to use pLDDT features
            b_angle: Whether to use angle features
            b_coord: Whether to use coordinate features
            lora_r: LoRA rank
            peft: Whether to use PEFT (Parameter-Efficient Fine-Tuning)
        """
        super().__init__()
        
        # Initialize basic parameters
        self.num_labels = num_labels
        self.task = task
        self.config_path = config_path
        self.b_angle = b_angle
        self.b_coord = b_coord
        self.fusion_dim = fusion_dim
        self.loss_fn = loss_fn
        
        # Set feature dimension based on model size
        self.feat_dim = {
            '650': 1280,
            '35': 480,
            'saprot': 1280
        }.get(model_size, 1280)
        
        # Initialize LoRA configuration
        lora_config = {
            "r": lora_r,
            "target_modules": ["query", "key", "value", "intermediate.dense", "output.dense"],
            "modules_to_save": ["self.cupdate"],
            "inference_mode": False,
            "lora_dropout": 0.1,
            "lora_alpha": 16,
        }
        
        # Initialize tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config_path)
        self.esm_model = EsmModel.from_pretrained(self.config_path)
        
        # Apply PEFT if enabled
        if peft:
            peft_config = LoraConfig(**lora_config)
            self.esm_model = get_peft_model(self.esm_model, peft_config)
        
        # Initialize model components
        self._init_components(plddt)
        self._init_loss_function(loss_fn)
        self._init_geometric_components()
        
        print_trainable_parameters(self.esm_model)
    
    def _init_components(self, plddt):
        """Initialize model components."""
        self.gamma = nn.Parameter(torch.tensor(0.0000001))
        self.mlp = EsmClassificationHead(self.feat_dim * 2, self.num_labels)
        self.lbs_head = EsmClassificationHead(self.feat_dim, self.num_labels)
        
        if plddt:
            self.fusion = Fusion_plddt(self.feat_dim, self.fusion_dim)
        else:
            self.fusion = Fusion(self.feat_dim, self.fusion_dim)
    
    def _init_loss_function(self, loss_fn):
        """Initialize loss function."""
        if loss_fn == 'BCE':
            self.loss_ce = nn.BCEWithLogitsLoss()
        elif loss_fn == 'MSE':
            self.loss_ce = nn.MSELoss()
        elif loss_fn == 'CE':
            self.loss_ce = nn.CrossEntropyLoss()
    
    def _init_geometric_components(self):
        """Initialize geometric processing components."""
        self.cutoff = 16
        self.num_rbf = 50
        self.embedding_dim = 20
        self.act = nn.SiLU()
        
        self.distance_expansion = ExpNormalSmearing(self.cutoff, self.num_rbf)
        self.dist_proj = nn.Linear(self.num_rbf, self.embedding_dim)
        
        # Initialize weights
        self.distance_expansion.reset_parameters()
        nn.init.xavier_uniform_(self.dist_proj.weight)
        self.dist_proj.bias.data.fill_(0.0)
    
    def dist_acculate(self, coords, attention_mask):
        """Calculate distance-based features."""
        B, N, _ = coords.shape
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        
        # Create masks
        pos_mask = ~(attention_mask.unsqueeze(1) | attention_mask.unsqueeze(2))
        loop_mask = torch.eye(N, dtype=torch.bool, device=coords.device)
        loop_mask = repeat(loop_mask, "n m -> b n m", b=B)
        
        # Calculate distances
        dist = torch.norm(coords.unsqueeze(1) - coords.unsqueeze(2), dim=-1)
        dist = dist.masked_fill(loop_mask, 0.0)
        
        # Process edge attributes
        adj_mask = (dist < self.cutoff) & pos_mask
        edge_attr = self.distance_expansion(dist)
        edge_attr = self.dist_proj(edge_attr)
        edge_attr = self.act(edge_attr)
        edge_attr = edge_attr.masked_fill(~adj_mask.unsqueeze(-1), 0.0)
        
        return dist, edge_attr
    
    def forward(
        self,
        inputs,
        ids,
        attention_mask,
        label,
        coords,
        plddt=False,
        lens=None
    ):
        """Forward pass."""
        bs, n = ids.shape
        
        # Process angle features if enabled
        if self.b_angle:
            if plddt:
                angle = self.fusion(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
            else:
                angle = self.fusion(inputs[0], inputs[1], inputs[2], inputs[3])
            
            zeros = torch.zeros(bs, 1, self.feat_dim).to(ids.device)
            angle = torch.cat((zeros, angle, zeros), dim=1)
        
        # Process coordinate features if enabled
        if self.b_coord:
            zeros = torch.zeros(bs, 1, coords.shape[-1]).to(ids.device)
            coords = torch.cat((zeros, coords, zeros), dim=1)
            dist, edge_attr = self.dist_acculate(coords, attention_mask)
        
        # Get model output based on enabled features
        x_vis = self._get_model_output(
            ids=ids,
            attention_mask=attention_mask,
            angle=angle if self.b_angle else None,
            dist=dist if self.b_coord else None,
            edge_attr=edge_attr if self.b_coord else None
        )
        
        # Task-specific processing
        if self.task == 'lbs':
            return self._process_lbs_task(x_vis, label, lens)
        else:
            return self._process_default_task(x_vis, label)
    
    def _get_model_output(self, ids, attention_mask, angle, dist, edge_attr):
        """Get model output based on enabled features."""
        kwargs = {
            'input_ids': ids,
            'attention_mask': attention_mask,
            'gamma': self.gamma
        }
        
        if self.b_angle and angle is not None:
            kwargs['angle'] = angle
        
        if self.b_coord and dist is not None and edge_attr is not None:
            kwargs['structure'] = (dist, edge_attr)
        
        return self.esm_model(**kwargs)['last_hidden_state']
    
    def _process_lbs_task(self, x_vis, label, lens):
        """Process LBS (Local Binding Site) task."""
        ret = self.lbs_head(x_vis)[:, 1:-1]
        label = label[:, :ret.shape[1]]
        loss = self.loss_ce(ret, label.float())
        acc = 1 - loss
        return ret, loss, acc
    
    def _process_default_task(self, x_vis, label):
        """Process default classification/regression task."""
        concat_f = torch.cat((x_vis.mean(1), x_vis[:, 1:].max(1)[0]), dim=1)
        
        if self.num_labels == 1 and self.task == 'classification':
            ret = self.mlp(concat_f)[:, 0]
        else:
            ret = self.mlp(concat_f)
        
        if self.loss_fn == 'CE':
            loss = self.loss_ce(ret, label.long())
        else:
            loss = self.loss_ce(ret, label)
        
        return ret, loss, self.gamma