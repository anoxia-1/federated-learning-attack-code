# 关于联邦学习攻击预防与的论文代码复现
## 已复现的相关论文如下
* How To Backdoor Federated Learning
* A3FL Adversarially Adaptive Backdoor Attacks to Federated Learning
* Neurotoxin Durable Backdoors in Federated Learning
* DBA DISTRIBUTED BACKDOOR ATTACKS AGAINST FEDERATED LEARNING
* IBA Towards Irreversible Backdoor Attacks in Federated Learning

## 运行方式
### How To Backdoor Federated Learning
```
$ python training.py --params utils/params.yaml
```

### A3FL Adversarially Adaptive Backdoor Attacks to Federated Learning
```
$ python clean.py --gpu 0 --params configs/config.yaml
```

### Neurotoxin Durable Backdoors in Federated Learning
```
$ ./exps/mnist-multi-krum.sh
$ ./exps/cifar10-multi-krum.sh 
$ ./exps/timagenet-multi-krum.sh 
```
