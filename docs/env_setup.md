# Environment Setup


- We use conda to install required packages:
```
    conda env create -f env/env_equiformer_v2.yml
```

- This will create a new environment called `equiformer_v2`.

- We activate the environment:
```
    export PYTHONNOUSERSITE=True    # prevent using packages from base
    conda activate equiformer_v2
```

- Besides, [`env/env_equiformer_v2.yml`](../env/env_equiformer_v2.yml) specifies versions of all packages.

- After setting up the environment, clone OC20 GitHub repository:
```
    git clone https://github.com/FAIR-Chem/fairchem.git
    cd ocp
    git checkout f83d150
```

- The correpsonding version of OC20 GitHub repository is [here](https://github.com/FAIR-Chem/fairchem/tree/f83d150b5fcb940b814f25043fb00914945b8708).

- We need to modify `ocp/ocpmodels/common/utils.py` and add the following two lines after [Line 329](https://github.com/FAIR-Chem/fairchem/blob/f83d150b5fcb940b814f25043fb00914945b8708/ocpmodels/common/utils.py#L329) as shown below:
```diff
    finally:
+       import nets
+       import oc20.trainer
        registry.register("imports_setup", True)
```

- Finally, we install `ocpmodels` by running:
```
    # After activating the environment and under ocp/
    pip install -e .
```

