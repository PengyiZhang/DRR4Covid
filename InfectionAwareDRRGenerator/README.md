# InfectionAwareDRRGenerator

## Configuration in genDRRV2.py 

1. x-ray projector information
```
# 01: 1, 02: 0.5
focal_lenght = 1000
# Define projector for generation of DRR from 3D model (Digitally Reconstructed Radiographs)
projector_info = {'Name': 'SiddonGpu', 
                  'threadsPerBlock_x': 16,
                  'threadsPerBlock_y': 16,
                  'threadsPerBlock_z': 1,
                  'focal_lenght': focal_lenght,
                  'DRRspacing_x': 0.2756, # 0.5, 1
                  'DRRspacing_y': 0.2756,
                  'DRR_ppx': 3.6180,
                  'DRR_ppy': 3.6180,
                  'DRRsize_x': 1024,
                  'DRRsize_y': 1024,
                  }

```

2. Enabling the synthesis of multiple DRRs from a single CT volume by setting the pose of CT volume within virtual imaging system


```
def get_random_transform():
    transform_parameters = [0,0,0,0,0,0]
    index = random.randint(0,10000) % 6
    if index == 0:
        high_alpha_x, low_alpha_x = 45, -45
        alpha_x = random.randint(low_alpha_x, high_alpha_x) / 180.0 * 3.14
        transform_parameters = [alpha_x, 0, 0, 0, 0, 0]
    elif index == 1:
        high_alpha_z, low_alpha_z = 45, -45
        alpha_z = random.randint(low_alpha_z, high_alpha_z) / 180.0 * 3.14
        transform_parameters = [0, 0, alpha_z, 0, 0, 0]        
    elif index == 2:
        high_alpha_y, low_alpha_y = 30, 0
        alpha_y = random.randint(low_alpha_y, high_alpha_y) / 180.0 * 3.14
        transform_parameters = [0, alpha_y, 0, 0, 0, 0]
    elif index == 3: 
        transform_parameters = [0, 0, 0, random.randint(-100, 100), 0, 0]
    elif index == 4: 
        transform_parameters = [0, 0, 0, 0, random.randint(0, 100), 0]    
    else: 
        transform_parameters = [0, 0, 0, 0, 0, random.randint(-100, 100)]

    print(transform_parameters)

    return transform_parameters
```

3. The synthesis of normal DRRs by setting w0 = w1 = 24.0 and w2 = 1.0

```
def gen_drr_img():

    random.seed(2020)

    ClassWeights = [24.0, 24.0, 1.0]
    ...
```


4. CT volumes with annotation volumes
 

```
model_filepaths = ["../../data/volume/IMG/coronacases_%03d.nii.gz" % (i+1) for i in range(10)]

model_maskpaths = ["../../data/volume/GT/coronacases_%03d.nii.gz" % (i+1) for i in range(10)]

```

## the synthesis of infection-aware DRRs

```
python3.6 genDRRV2.py
```

