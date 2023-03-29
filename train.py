### training
for epoch in range(1, int(n_epochs)):

    GNN_core.train(model=model,train_loader=train_loader,optimizer=optimizer,criterion=criterion)
    
    print(epoch)
    break
    
    train_acc = GNN_core.test(model=model,loader=train_loader)
    test_acc = GNN_core.test(model=model,loader=test_loader)

    test_loss=GNN_core.loss(model=model,loader=test_loader,criterion=criterion).item()
    train_loss=GNN_core.loss(model=model,loader=train_loader,criterion=criterion).item()

    this_val_acc = GNN_core.test(model=model,loader=val_loader)
    if epoch %20==0:
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f},Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')

    if this_val_acc > best_val_acc: #validation wrapper
        best_val_epoch = epoch
        best_val_acc=this_val_acc
        best_model= copy.deepcopy(model)
        patience_counter = 0
        print(f"new best validation score {best_val_acc}")
    else:
        patience_counter+=1
    if patience_counter == patience:
        print("ran out of patience")
        break

trainscore = GNN_core.test(model=best_model,loader=train_loader)
testscore = GNN_core.test(model=best_model,loader=test_loader)
print(f'score on train set: {trainscore}')
print(f'score on test set: {testscore}')
predict_test = GNN_core.predict(model=best_model,loader=test_loader)
label_test=[]
for data in test_loader:
    label_test.append(data.y.tolist())

label_test=[item for sublist in label_test for item in sublist]
predict_test=[item for sublist in predict_test for item in sublist]
#predict_test=np.array(predict_test).ravel()

fpr1, tpr1, thresholds = roc_curve(label_test, predict_test)
tn, fp, fn, tp = confusion_matrix(label_test, predict_test).ravel()
AUROC = auc(fpr1, tpr1)
print(f'  AUC: {AUROC}')
print(f"  confusion matrix: [tn {tn}, fp {fp}, fn {fn}, tp {tp}]")
precision = tp/(tp+fp)
recall = tp/(tp+fn)
print(f'  precision = {precision}')
print(f'  recall = {recall}')
# print(args)
print(round(AUROC,3),round(trainscore,3),round(testscore,3),round(precision,3),round(recall,3),tn, fp, fn, tp)


# save model
best_model_wts = copy.deepcopy(best_model.state_dict())
save_path = os.path.join('.','{}_best_model.pth'.format(arch))
print(save_path)
torch.save(best_model_wts, save_path)
