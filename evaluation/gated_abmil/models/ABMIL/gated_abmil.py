import torch.nn as nn
import torch
import numpy as np
import os
from loguru import logger
from tqdm import tqdm

class GatedABMIL(nn.Module):

    def __init__(self, emb_dim, hidden_dim, num_heads=1, feature_extractor=None, classifier=None, learnable_values=False) -> None:
        super().__init__()

        self.V = nn.Linear(emb_dim, hidden_dim)
        self.U = nn.Linear(emb_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, num_heads)
 
        if learnable_values:
            self.value_proj = nn.Linear(emb_dim, emb_dim)
        else:
            self.value_proj = nn.Identity()
        
        self.num_heads = num_heads

        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = nn.Identity()

        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = nn.Identity()

    def forward(self, x, mask=None):
        """
        x: input of size B x S x D
        mask: mask of size B x S indicating padding (0)
        """
        x = self.feature_extractor(x)

        v_x = self.V(x)
        u_x = self.U(x)

        v_x = nn.functional.tanh(v_x)
        u_x = nn.functional.sigmoid(u_x)

        h = v_x * u_x

        attn_scores = self.W(h) #B x S x H
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(2), -1e9)
        attn_weights = nn.functional.softmax(attn_scores, dim=1) # B x S x H
        attn_weights = attn_weights.transpose(1, 2) # B x H x S

        output = torch.bmm(attn_weights, self.value_proj(x)) # B x H x D
        output_flat = output.reshape(-1, self.num_heads * x.size(2))

        return self.classifier(output_flat), output_flat

    def compute_attention(self, x, mask=None, batched=True):
        """
        x: input of size B x S x D
        """
        if not batched:
            x = x.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        if x.dim() > 3:
            reshaped = True
            old_shape = x.shape
            x = x.reshape(x.size(0), -1, x.size(-1))
        else:
            reshaped = False

        x = self.feature_extractor(x)
        v_x = self.V(x)
        u_x = self.U(x)

        v_x = nn.functional.tanh(v_x)
        u_x = nn.functional.sigmoid(u_x)

        h = v_x * u_x

        attn_scores = self.W(h) #B x S x H
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)
        attn_weights = nn.functional.softmax(attn_scores, dim=1) # B x S x H
        attn_weights = attn_weights.transpose(1, 2) # B x H x S
        
        if reshaped:
            attn_weights = attn_weights.reshape(old_shape[0], self.num_heads, *old_shape[1:-1])
        if not batched:
            attn_weights = attn_weights.squeeze(0)

        return attn_weights


class GatedABMILClassifierWithValidation(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_heads=1, num_classes=2, patience=5, verbose=True, save_path="src/abmil_factory/expt/", name=None, device='cuda'):
        super().__init__()
        
        self.num_classes = num_classes

        if self.num_classes == 2:
            classifier = nn.Sequential(
                nn.Linear(num_heads * input_dim, self.num_classes-1),
            )
            
        else:
            classifier = nn.Sequential(
                nn.Linear(num_heads * input_dim, self.num_classes),
            )

        self.patience = patience
        self.patience_counter = 0
        self.model = GatedABMIL(input_dim, hidden_dim, num_heads=num_heads, classifier=classifier)
        self.verbose = verbose

        self.best_model = None

        self.save_path = save_path
        self.name = name
        
        if self.name is None:
            self.name = "gated_abmil"


        if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
            if verbose:
                logger.warning("CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        self.device = device    

    def forward(self, x, mask=None):
        return self.model(x, mask=mask)

    def compute_attention(self, x, mask=None, batched=True):
        return self.model.compute_attention(x, mask=mask, batched=batched)
    

    @torch.no_grad()
    def valid_eval(self, valid_dl, loss_fn):
        self.eval()
        valid_loss = 0
        predictions = []
        ground_truth = []
        for batch in valid_dl:
            bags, masks, labels = batch
            bags = bags.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            logits, _ = self.forward(bags, mask=masks)
            # if self.num_classes == 2:
            #     labels = labels.unsqueeze(1).float()

            # loss = loss_fn(logits, labels)

            if self.num_classes == 2:
                labels = labels.unsqueeze(1).float()
            # print(logits.shape, labels.shape)
            loss = loss_fn(logits, labels)
            if self.num_classes > 2:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = torch.sigmoid(logits).round().squeeze(1)
            
            predictions.append(preds.cpu().numpy())
            ground_truth.append(labels.cpu().numpy())

            valid_loss += loss.item()

            # print(logits)

        predictions = np.concatenate(predictions)
        ground_truth = np.concatenate(ground_truth)

        if self.num_classes > 2:
            accuracy = (predictions == ground_truth).mean()
        else:
            accuracy = (predictions == ground_truth.squeeze(-1)).mean()

        # accuracy = (predictions == ground_truth).mean()

        return valid_loss / len(valid_dl), accuracy
    

    @torch.no_grad()
    def get_outputs(self, valid_dl, loss_fn=None, load_best=True):

        if load_best:
            self.load_best_model()

        self.eval()
        valid_loss = 0
        outputs = []
        ground_truth = []
        predictions = []
        for batch in valid_dl:
            bags, masks, labels = batch
            bags = bags.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            logits, output_flat = self.forward(bags, mask=masks)
          
            if self.num_classes == 2:
                labels = labels.unsqueeze(1).float()
            if self.num_classes > 2:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = torch.sigmoid(logits).round().squeeze(1)
            
            predictions.append(preds.cpu().numpy())
            outputs.append(logits.cpu().numpy())
            ground_truth.append(labels.cpu().numpy())

        predictions = np.concatenate(predictions)
        ground_truth = np.concatenate(ground_truth)

        outputs = np.concatenate(outputs)

        if self.num_classes == 2:
            ground_truth = ground_truth.squeeze(-1)

        return {
            'logits': outputs,
            'ground_truth': ground_truth,
            'predictions': predictions,
        }


    def train_model(self, train_dl, valid_dl, num_epochs, optimizer, loss_fn, test_dl=None, monitor="valid_loss"):
        """
        Trains the model. train_dl should return batches of the form (bags, masks, labels) where bags BxSxD, masks BxS, labels B
        """
        self.to(self.device)
        self.average_train_loss = []
        self.average_train_accuracy = []
        self.average_valid_loss = []
        self.average_valid_accuracy = []

        if test_dl is not None:
            self.average_test_loss = []
            self.average_test_accuracy = []

        
        self.best_val_acc = 0
        self.best_val_loss = np.inf


        for ep in (range(num_epochs)):
            train_loss = 0
            self.train()
            for batch in train_dl:
                bags, masks, labels = batch
                bags = bags.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                logits, _ = self.forward(bags, mask=masks)
                if self.num_classes == 2:
                    labels = labels.unsqueeze(1).float()
                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            self.average_train_loss.append(train_loss / len(train_dl))

            valid_loss, valid_accuracy = self.valid_eval(valid_dl, loss_fn)
            self.average_valid_accuracy.append(valid_accuracy)
            self.average_valid_loss.append(valid_loss)

            if test_dl is not None:
                test_loss, test_accuracy = self.valid_eval(test_dl, loss_fn)
                self.average_test_accuracy.append(test_accuracy)
                self.average_test_loss.append(test_loss)
                if self.verbose:
                    logger.info(f"Epoch: {ep} Train Loss: {train_loss / len(train_dl):.3f}, Valid Loss: {valid_loss:.3f}, Valid Accuracy: {valid_accuracy:.3f}, Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}")
            else:
                if self.verbose:
                    logger.info(f"Epoch: {ep} Train Loss: {train_loss / len(train_dl):.3f}, Valid Loss: {valid_loss:.3f}, Valid Accuracy: {valid_accuracy:.3f}")
                

            if monitor == "valid_accuracy":

                if valid_accuracy > self.best_val_acc:
                    self.best_val_acc = valid_accuracy
                    self.best_val_loss = valid_loss
                    self.best_model = self.model.state_dict().copy()

                    os.makedirs(self.save_path, exist_ok=True)
                    torch.save(self.best_model, f"{self.save_path}/{self.name}.pt")

                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

            elif monitor == "valid_loss":
                if valid_loss < self.best_val_loss:
                    self.best_val_acc = valid_accuracy
                    self.best_val_loss = valid_loss
                    self.best_model = self.model.state_dict().copy()
                    os.makedirs(self.save_path, exist_ok=True)
                    torch.save(self.best_model, f"{self.save_path}/{self.name}.pt")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            
            else:
                raise ValueError("monitor must be either valid_accuracy or valid_loss")
            
            if self.patience_counter >= self.patience:

                logger.info(f"Early stopping with best {monitor} @ loss: {self.best_val_loss:.3f}, acc: {self.best_val_acc:.3f}")
                break

    def load_best_model(self):
        self.model.load_state_dict(torch.load(f"{self.save_path}/{self.name}.pt"))