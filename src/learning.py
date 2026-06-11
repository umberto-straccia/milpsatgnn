from collections import defaultdict
import gc
import humanize as H
import json
import numpy as np
import os
from os.path import join
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import psutil
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torchinfo import summary
from tqdm import tqdm
import yaml

import learners.common as LC

from data.data_utils import (
    load_dataset_splits, load_dataset_splits_g4s
  )  

def do_train(conf, run):
    model = None
    dev = conf["dev_mode"]
    # set visible devices
    device = set_device(conf)
    
    print(f"Using device: {device}")
    print("Reading dataset...")
    if conf.get("external_data", None) is not None:
        if conf.external_data == "g4satbench":
            print("Using G4SATBench dataset")
            splits = load_dataset_splits_g4s(conf, ["train", "val"])
        else:
            assert False, f"Unknown external source: {conf.external_data}"
    else:
        splits = load_dataset_splits(conf, ["train", "val"])

    print("TRAINING size:", len(splits["train"]))
    print("VALIDATION size:", len(splits["val"]))

    example = splits["train"][0]
    n_cons_f = example["left"].x.size(1)
    n_var_f = example["right"].x.size(1)
    

    if dev:
        print( "TYPE:", type(example) )
        print( "EXAMPLE [0]:", example )
        example = splits["train"][1]
        print( "EXAMPLE [1]:", example )
        print("--")
        print("constraint, n features:", n_cons_f)
        print("variable, n features:", n_var_f)
    
    print(f"RAM memory:")
    m = psutil.virtual_memory()
    print(f"Total / Available: {H.naturalsize(m.total)} / {H.naturalsize(m.available)}")
    
    # Create data loaders
    train_loader = DataLoader(splits["train"], batch_size=int(conf["batch_size"]), shuffle=True, ) # True)
    val_loader = DataLoader(splits["val"], batch_size=int(conf["batch_size"]), shuffle=False)

    if dev:
        batch = next(iter(train_loader))
        print(">>")
        print(f"\tDEV, bs={conf['batch_size']}, first batch:")
        print(batch)
        print("<<")

    print("WARNING: 'model name' in configuration ignored - initializing GCNPolicy")
    # params: emb_size, n_cons_f, n_edge_f, n_var_f, n_layers=2,
    model = LC.build_model(conf, cons_dim=n_cons_f, vars_dim=n_var_f)
    model.print_info()
    
    if conf.get("weight_decay", None) is not None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(conf["lr"]), weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(conf["lr"])) # , weight_decay=1e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=float(conf["lr"]), weight_decay=1e-4)
    
    if conf["use_lr_scheduler"]:
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01)
    else:
        scheduler = None
    # insert a learning rate scheduler with step size 10 and gamma 0.1
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    best_model_path = os.path.join(conf["output_dir"], conf["best_model_path"])
    final_model_path = os.path.join(conf["output_dir"], conf["final_model_path"])

    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

    t1 = time.perf_counter()
    dataset_path = os.path.join(conf["dataset_dir"], "pyg", conf.get("dataset", "<unspec>"))
    results = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        run=run,
        scheduler=scheduler,
        epochs=int(conf["epochs"]), 
        best_model_path = best_model_path, 
        final_model_path = final_model_path, 
        device = device, 
        best_with_loss = conf["best_with_loss"], 
        model_name="GCNPolicy",
        loss_function_name=conf["loss_function"],
        msgs = [f"Training dataset: {dataset_path}", f"device:{conf['device']}"],
        output_dir=conf["output_dir"],
      )  # conf["model"])
    t2 = time.perf_counter()
    results["training_time"] = t2 - t1
    results["training_time_human"] = H.precisedelta(t2 - t1)
    run["train/time"] = t2-t1
    run["train/time_human"] = H.precisedelta(t2 - t1)
    model = None
    gc.collect()

    # save train_loss
    # save json file with training results
    results_fn = join(conf["output_dir"], "training_results.json")
    with open(results_fn, "w") as f:
        json.dump(results, f, indent=4)
    run["files/training_results"].upload(results_fn)
    print("training info saved to:", results_fn)
#<

# target smoothing
def process_target(target):
    return 0.9 * target + 0.05  # torch.add(torch.multiply(target, 0.9), 0.05)

def train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer,
        run,
        scheduler=None,
        epochs=2, 
        best_model_path="best_model.pth", 
        final_model_path="final_model.pth", 
        device=torch.device("cpu"), 
        val_freq=1, 
        best_with_loss=False,
        dev=False,
        model_name="GCNPolicy",
        loss_function_name="mse",
        msgs=[],
        output_dir=None,
    ):
    
    model.to(device)
    
    # Use gradient clipping
    max_grad_norm = 1.0
    
    #prepare folders for saving models
    # os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    # os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    train_results = defaultdict(list)
    train_results["has_validation"] = len(val_loader) > 0

    # best_with_loss = True
    best_value = float('inf') if best_with_loss else float("-inf")

    best_metric_name = "accuracy"
    best_metric = accuracy_score
    
    
    best_epoch = -1
    
    # loss_fn = F.binary_cross_entropy
    # loss_fn = F.mse_loss
    # loss_fn = F.binary_cross_entropy

    if loss_function_name == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss_function_name == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        assert False, f"Unknown loss function: {loss_function_name}"
    
    num_batches = len(train_loader)
    
    train_loss = 0.0

    train_losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} - START")
        print("Training set:", len(train_loader.dataset))
        print("Validation set:", len(val_loader.dataset))
        print("Batch size:", train_loader.batch_size)
        print("Number of batches:", num_batches)
        print("Using loss:", loss_function_name)
        print("Using scheduler:", scheduler is not None)
        print("Learning rate: ", optimizer.param_groups[0]["lr"])
        for m in msgs:
            print(m)
        print()
        model.train()
        pbar = tqdm(train_loader, total=num_batches, position=0)
        
        # TODO: training samples should be shuffled

        running_loss = 0.0
        total_examples = 0

        for batch_idx, batch in enumerate(pbar):
            # print first example in batch
            if epoch == 0 and batch_idx == 0:
                print(batch[0])
            pbar.set_description(f"E {epoch+1:02d}/{epochs} - Best at {best_epoch:02d} ({best_metric_name} on val: {best_value:.3f}). Train loss: {train_loss:.3f}")
            optimizer.zero_grad()
            
            # print shapes
            if False:
                print(f"x.shape: {x.shape}")
                print(f"edge_index.shape: {edge_index.shape}")
                print(f"edge_attr.shape: {edge_attr.shape}")
                print(f"batch.shape: {batch.shape}")
                print(f"labels.shape: {labels.shape}")

            # Forward pass
            out = model(batch.to(device)).squeeze(-1)
            labels = batch.label.to(device)



            # loss = loss_fn(out, smoothed_target, reduction="mean") # use with CrossEntropyLoss
            if loss_function_name == "bce":
                loss = loss_fn(out, labels) # use with BCEWithLogitsLoss
            else:
                out = torch.sigmoid(out)
                loss = loss_fn(out, labels) # use with MSELoss
            #<
                
            # loss = nn.MSELoss(out, smoothed_target) # to be corrected, try this as a loss function
                
            # Backpropagation with gradient clipping
            # print("Loss shape:", loss.shape)
            # print(loss)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            running_loss += ( loss.item() * len(batch) )
            total_examples += len(batch)
            
        #< training loop: for over batches
        train_loss = running_loss / total_examples
        train_losses.append(train_loss)
        run["train/loss"].append(step=epoch, value=train_loss)
        
        if (epoch+1) % 5 == 0:
            train_results["train_validation_epochs"].append(epoch)
            train_preds, _ = make_predictions(model, train_loader, device, loss_fn=None)
            train_metrics, train_cm = compute_metrics(pd.DataFrame.from_dict(train_preds))
            train_acc = train_metrics["accuracy"]
            for k, v in train_metrics.items():
                tqdm.write(f"(prediction on training set) \t-  {k}: {v}")
                train_results[f"train/{k}"].append( v )
                run[f"train/{k}"].append(step=epoch, value=v)
        # train_loss = train_loss / num_batches
        else:
            train_acc = None
        
        save_model = False
        if (len(val_loader) == 0) and (train_loss < best_value):
            print("No validation set - choosing best model with train loss")
            best_value = train_loss
            best_epoch = epoch+1
            save_model = True
            assert False, "Never here"
        #<

        if (len(val_loader)>0) and ( (epoch+1) %  val_freq == 0):
            train_results["validation_epochs"].append(epoch)

            # print(f"Epoch {epoch+1}/{epochs} - VALIDATION")
            res_dict, losses = make_predictions(model, val_loader, device, loss_fn, loss_function_name)
            
            res_df = pd.DataFrame.from_dict(res_dict)
            val_loss = np.mean(losses)
            train_results["validation_loss"].append( val_loss )
            run["val/loss"].append(step=epoch, value=val_loss)
            val_metrics, val_cm = compute_metrics(res_df)
            if scheduler is not None:
                if scheduler.mode == "min":
                    scheduler.step(val_loss)
                else:
                    # TODO: accuracy is recomputed below. use val_metrics
                    scheduler.step(val_metrics["accuracy"])
            
            print(pd.DataFrame(val_cm, columns=["Pred 0", "Pred 1"], index=["0", "1"]).to_markdown())
            tqdm.write("* VALIDATION METRICS")
            for k, v in val_metrics.items():
                tqdm.write(f"\t-  {k}: {v}")
                train_results[k].append( v )
                run[f"val/{k}"].append(step=epoch, value=v)
            if not best_with_loss:
                val_acc = best_metric(res_dict["label"], res_dict["prediction"])
                if val_acc > best_value:
                    best_value = val_acc
                    best_epoch = epoch+1
                    save_model = True
                    # XXX
                    run["val/current_best"].append(step=epoch, value=val_acc)
                # XXX
                run["val/running_best"].append(step=epoch, value=val_acc)
                
            elif val_loss <= best_value:
                best_value = val_loss
                best_epoch = epoch+1
                save_model = True
                
            
        #< end validation
        if False and (train_acc is not None) and (epoch>50) and (train_acc <= 0.5):
            print("Stopping training - accuracy is 0.5")
            if output_dir is not None:
                with open(os.path.join(output_dir, "warning.txt"), "w") as fout:
                    fout.write(f"Stopping training at epoch {epoch} - accuracy is {train_acc}")
            break

        if save_model:
            print(f"New best at epoch {best_epoch} - using loss? {best_with_loss}")
            torch.save(model.state_dict(), best_model_path)
        #<

        print()
    #< for over epochs

    # Save final model
    torch.save(model.state_dict(), final_model_path)

    res = {}
    res["training_loss"] = train_losses
    res["best_with_loss"] = best_with_loss
    res["best_metric"] = str(best_metric)
    res["best_epoch"] = best_epoch
    res["best_value"] = best_value
    res["best_model_path"] = best_model_path
    res["final_model_path"] = final_model_path
    res["last_epoch"] = epoch
    for k, v in train_results.items():
        res[k] = v
    return res
#< train_model



def set_device(conf):
    if "cuda_devices" in conf.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] = map(int, conf["cuda_devices"].split(","))
        print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])

    # set torch device to conf["device"] if cuda is available
    
    if "device" in conf.keys():
        comp_device =  conf["device"] if torch.cuda.is_available() else "cpu"
        device = torch.device(comp_device)
    else:
        device = torch.device("cpu")
    return device
#<

    

def make_predictions(model, dataloader, device, loss_fn=None, loss_fn_name=None):
    model.eval()
    
    outs = []
    out_labels = []
    preds = []
    formula_ids = []
    losses = []

    #val_pbar = tqdm(dataloader, desc="Predicting", position=2, leave=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting")):
            out = model(batch.to(device)).squeeze(-1)
            labels = batch.label.to(device)

            if loss_fn is not None:
                assert loss_fn_name is not None, "loss_fn_name must be provided if loss_fn is provided"
                if loss_fn_name == "bce":
                    l = loss_fn(out, labels).item()
                else:
                    l = loss_fn(torch.sigmoid(out), labels).item()
                losses.append(l)
            #<
            
            
            out = F.sigmoid(out)
            out = out.cpu().numpy().flatten()
            outs.extend(out)

            pred_feasible = (out.squeeze() > 0.5).astype(int)
            preds.extend(pred_feasible)            
            labs = labels.cpu().numpy().astype(int)
            out_labels.extend(labs)
            
            # for f, o, feas, l in zip(features.formula_id, out, pred_feasible, labs):
            #     print(f"FI: {f}, o: {o:.3f}, p: {feas}, l: {int(l.item())}")
            formula_ids.extend(batch["formula_id"].cpu().numpy())
        #< for over batches

    # df = pd.DataFrame({"formula_id": formula_ids, "out": outs, "prediction": preds, "label": out_labels})
    res_dict = {
        "formula_id": formula_ids,
        "out": outs,
        "prediction": preds,
        "label": out_labels
    }
    return res_dict, losses


def do_test(conf, run):
    model = None

    model_path = conf["best_model_path"] if conf["test_with_best"] else conf["final_model_path"]
    model_path = join(conf["output_dir"], model_path)
    if not os.path.exists(model_path):
        print("No model found at the specified path:", model_path)
        exit(1)
    #<
    device = set_device(conf)        
    
    # test_dataset, labels = load_dataset(test_path, testing= True)
    if conf.get("external_data", None) is not None:
        if conf.external_data == "g4satbench":
            print("Using G4SATBench dataset")
            test_split = load_dataset_splits_g4s(conf, ["test"])["test"]
        else:
            assert False, f"Unknown external source: {conf.external_data}"
    else:
        test_split = load_dataset_splits(conf, ["test"])["test"]
    
    test_loader = DataLoader(test_split, batch_size=int(conf["test_batch_size"]), shuffle=False)
    
    example = test_split[0]
    n_cons_f = example["left"].x.size(1)
    n_var_f = example["right"].x.size(1)
    
    model = LC.build_model(conf, cons_dim=n_cons_f, vars_dim=n_var_f)
    
    model.print_info()

    model.load_state_dict(torch.load(model_path , weights_only=True))
    model.to(device)
    time1 = time.perf_counter()
    res_df = test_model(model, test_loader, device)
    time2 = time1 = time.perf_counter()
    
    avg_time_per_ex = (time2 - time1) / len(test_loader.dataset)
    print(f"Average time per example: {avg_time_per_ex:.4f} s")
    res_df["test_t"] = time2 - time1
    res_df["avg_test_t"] = avg_time_per_ex
    out_fn = join(conf["output_dir"], f"test-examples-predictions.csv")
    res_df.to_csv(out_fn, index=False, sep=",")
    print(f"Results saved to {out_fn}")
    
    metrics_results, test_cm = compute_metrics(res_df) 
    
    
    print("RESULTS:")
    for k, v in metrics_results.items():
        print(f"\t{k}: {v}")
        run[f"test/{k}"] = v
    
    out_fn = join(conf["output_dir"], f"test-results.csv")
    # save dict to csv file
    # dump metrics_results as yaml
    with open(out_fn, "w") as f:
        yaml.dump(metrics_results, f, indent=4)
    # metrics_results.to_csv(out_fn, index=False, sep=",")
    run["files/test_predictions"].upload(out_fn)
    print(f"Results saved to {out_fn}")  
#<

def test_model(model, test_loader, device):
    model.eval()
    
    n_examples = len(test_loader.dataset)
    print(f"Testing on {n_examples} examples")
    
    res_dict, _ = make_predictions(model, test_loader, device)
    df = pd.DataFrame.from_dict(res_dict)
    # df = pd.DataFrame({'formula_id': formula_ids, 'prediction': predictions, 'label': labels})
    df["correct"] = df["prediction"] == df["label"]
    df["tp"] = (df["prediction"] == 1) & (df["label"] == 1)
    df["tn"] = (df["prediction"] == 0) & (df["label"] == 0)
    df["fp"] = (df["prediction"] == 1) & (df["label"] == 0)
    df["fn"] = (df["prediction"] == 0) & (df["label"] == 1)
    for col in ["correct", "tp", "tn", "fp", "fn"]:
        df[col] = df[col].astype(int)
    # df["avg_t"] = avg_time_per_ex
    
    return df
#<

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def compute_metrics(df):
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score
    }
    
    y_true = df["label"]
    y_pred = df["prediction"]

    cm = confusion_matrix(y_true, y_pred)

    res = {}
    for k, metric in metrics.items():
        res[k] = metric(y_true, y_pred)

    return res, cm
#<
