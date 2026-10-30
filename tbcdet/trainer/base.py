from torch.utils.data import DataLoader

class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 epochs: int,
                 train_dataloader: DataLoader,
                 cls_loss,
                 bbox_loss,):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.cls_loss = cls_loss
        self.bbox_loss = bbox_loss
        
    def train(self):
        self.model.train()
        for i in range(1, self.epochs + 1):
            for inputs, labels in self.train_dataloader:
                inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
                cls_score, bbox_pred = self.model(inputs)
                
                cls_loss = self.cls_loss(labels, cls_score)
                reg_loss = self.bbox_loss(labels, bbox_pred)
