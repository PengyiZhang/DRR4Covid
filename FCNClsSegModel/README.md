# FCNClsSegModel

## training

1. train with using our domain adaptation module

```
python3.6 train_mmd.py --config cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-50.yaml --fold 0 --setgpuid 0
```

2. train without using our domain adaptation module

```
python3.6 train_no_mmd.py --config cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-50.yaml --fold 0 --setgpuid 0
```

## evaluation

1. Evaluate the model trained by using our domain adaptation module

```
python3.6 eval_mmd.py --config cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-50.yaml --fold 0 --setgpuid 0
```

2. Evaluate the model trained by not using our domain adaptation module

```
python3.6 eval_no_mmd.py --config cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-50.yaml --fold 0 --setgpuid 0
```



