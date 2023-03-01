Installation

```
conda create -n organ_seg python
conda activate organ_seg
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

Also install the packages under `../../src` with `pip install .` in each.

Those are:
  github.com/funkelab/funlib.learn.torch
  github.com/funkelab/funlib.persistence
  github.com/funkelab/funlib.show.neuroglancer
  github.com/funkey/gunpowder
