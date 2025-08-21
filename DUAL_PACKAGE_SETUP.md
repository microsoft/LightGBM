# LightGBM-Blinder: Dual Package Setup

## What's Been Done

✅ **Package Configuration Updated**
- Changed package name to `lightgbm-blinder` for pip installation
- Added version suffix `-blinder` to distinguish from original
- Updated repository URLs to point to your fork
- Configured build system for proper compilation

✅ **Side-by-Side Installation Working**
- Original LightGBM: `pip install lightgbm` (version 3.3.5)
- Your blinding fork: `pip install -e python-package` (version 4.5.0-blinder)
- Both packages can coexist without conflicts

✅ **Repository Cleaned Up**
- Removed all backup files, test scripts, and temporary files
- Committed only the meaningful package configuration changes
- Working directory is clean

## How to Use

### Original LightGBM (Standard ML)
```python
import lightgbm as lgb
# Use normally - no blind_volume parameter
```

### Your Blinding Fork (Privacy-Preserving ML)
```python
import sys
sys.path.insert(0, '/home/_/ATOL/LightGBM-Blinder/python-package')
if 'lightgbm' in sys.modules:
    del sys.modules['lightgbm']
import lightgbm as lgb_blind

# Use with blind_volume parameter
params = {'blind_volume': 0.3, ...}
```

## Future Updates

To merge changes from upstream LightGBM:
```bash
git remote add upstream https://github.com/microsoft/LightGBM.git
git fetch upstream
git merge upstream/master
pip install -e python-package --force-reinstall
```

Your fork is now ready for production use! 🎉
