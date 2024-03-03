import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from kornia.losses import focal_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from GCN.feeder.feeder_ABC import Feeder
from GCN.models.gcn_model import Model
from GCN.utils.AverageMeter import AverageMeter


def train():
  #Model
  model = Model(in_channels, num_class).to(device)

  #Loss & Optimizer
  optimizer = optim.Adam(
      model.parameters(),
      lr=base_lr,
      weight_decay=weight_decay
  )
  scheduler = optim.lr_scheduler.StepLR(
      optimizer,
      step_size=scheduler_step_size,
      gamma=0.01
  )

  criterion = nn.CrossEntropyLoss()

  #Meters
  train_loss_meter = AverageMeter()
  val_loss_meter = AverageMeter()
  acc_meter = AverageMeter()
  precision_meter = AverageMeter()
  recall_meter = AverageMeter()
  f1_score_meter = AverageMeter()

  # Dataloader
  dataset = Feeder(root_path)

  train, val = train_test_split(
      dataset,
      test_size=0.3
  )

  train_loader = DataLoader(
      train,
      batch_size=128,
      shuffle=True,
  )

  vald_loader = DataLoader(
      val,
      batch_size=128,
      shuffle=False,
  )
  print("Number sample of train dataset: ", len(train))
  print("Number sample of val dataset: ", len(val))

  best_f1 = 0
  stale = 0
  start_epoch = 1

  training_losses = []  # List to store training losses for each epoch
  validation_results = []

  for epoch in range(start_epoch, 1+MAX_EPOCHS):
      #Start time
      start_time = time.time()
      #Train
      model.train()
      #Reset meters
      train_loss_meter.reset()
      precision_meter.reset()
      recall_meter.reset()
      f1_score_meter.reset()
      acc_meter.reset()

      for batch_idx, (data, label, index) in enumerate(train_loader):
          n = data.shape[0]
          optimizer.zero_grad()
          data = data.float().to(device)
          label = label.long().to(device)

          output = model(data)

          train_loss = criterion(output, label)

          train_loss.backward()

          optimizer.step()
          scheduler.step()

          train_loss_meter.update(train_loss.item(),n)

      end_time = time.time()
      print(f"Training Result: Epoch {epoch}/{MAX_EPOCHS}, Loss: {train_loss_meter.avg:.3f}, Time epoch: {end_time-start_time:.3f}s")

      #Valid
      model.eval()
      with torch.no_grad():
          for batch_idx, (data, label, index) in enumerate(vald_loader):
              n = data.shape[0]
              data = data.float().to(device)
              label = label.long().to(device)

              output = model(data)
              val_loss = criterion(output, label)

              #Calculate metrics
              #P, R and F1
              label = label.detach().cpu().numpy()
              output = output.argmax(1).detach().cpu().numpy()

              p_score = precision_score(label, output, average='macro', zero_division=0)
              r_score = recall_score(label, output, average='macro', zero_division=0)
              _f1_score = f1_score(label, output, average='macro')
              acc = accuracy_score(label, output)

              #Update meters
              val_loss_meter.update(val_loss.item(), n)
              acc_meter.update(acc.item(),n)
              precision_meter.update(p_score.item(), n)
              recall_meter.update(r_score.item(), n)
              f1_score_meter.update(_f1_score.item(), n)

      print(f"Validation Result: Loss: {val_loss_meter.avg:.3f}, Accuracy: {acc_meter.avg:.3f} F1-Score: {f1_score_meter.avg:.3f}, Precision: {precision_meter.avg:.3f}, Recall: {recall_meter.avg:.3f}")

      #Save best model
      to_save = {
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'best_f1': best_f1,
          }
      if f1_score_meter.avg > best_f1:
          print(f"Best model found at epoch {epoch}, saving model")
          torch.save(to_save, os.path.join(weight_dir,f"best_epoch{epoch}_f1={f1_score_meter.avg:.3f}.pth"))
          best_f1 = f1_score_meter.avg
          stale = 0
      else:
          stale += 1
          if stale > 300:
              print(f"No improvement {300} consecutive epochs, early stopping")
              break
      if epoch % SAVE_INTERVAL == 0 or epoch == MAX_EPOCHS:
          print(f"Save model at epoch {epoch}, saving model")
          torch.save(to_save, os.path.join(weight_dir,f"epoch_{epoch}.pth"))

      # Store results for visualization
      training_losses.append(train_loss_meter.avg)
      validation_results.append({
          'epoch': epoch,
          'val_loss': val_loss_meter.avg,
          'accuracy': acc_meter.avg,
          'f1_score': f1_score_meter.avg,
          'precision': precision_meter.avg,
          'recall': recall_meter.avg
      })

  return training_losses, validation_results

if __name__ == '__main__':
    root_path = '/Users/vibuitruong/Documents/GitHub/ABC_Challenge_2024_bun_bo_team/data/Dataset-2'
    weight_dir = '/Users/vibuitruong/Documents/GitHub/ABC_Challenge_2024_bun_bo_team/GCN/output'
    in_channels = 2
    num_class = 9
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_decay = 0.00001
    base_lr = 0.0001
    MAX_EPOCHS = 10
    SAVE_INTERVAL = 10
    scheduler_step_size = MAX_EPOCHS * 0.6

    training_losses, validation_results = train()

