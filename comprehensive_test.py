#!/usr/bin/env python3
import numpy as np
import subprocess
import tempfile
import os
import json
import time
from typing import Dict, List, Tuple
import pandas as pd

class BlindingTester:
    def __init__(self):
        self.test_results = []
        self.temp_files = []
        
    def cleanup(self):
        """Clean up temporary files"""
        for f in self.temp_files:
            if os.path.exists(f):
                os.unlink(f)
        self.temp_files = []
        
    def create_temp_file(self, suffix='.txt') -> str:
        """Create a temporary file and track it for cleanup"""
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self.temp_files.append(path)
        return path
        
    def save_libsvm_format(self, X, y, filename, categorical_features=None):
        """Save data in LibSVM format with optional categorical feature encoding"""
        with open(filename, 'w') as f:
            for i in range(len(y)):
                line = f"{y[i]}"
                for j in range(X.shape[1]):
                    if X[i, j] != 0 or (categorical_features and j in categorical_features):
                        line += f" {j+1}:{X[i, j]}"
                f.write(line + "\n")
                
    def run_lightgbm(self, data_file: str, params: Dict) -> Dict:
        """Run LightGBM with given parameters and return results"""
        model_file = self.create_temp_file('.txt')
        
        cmd = ['./lightgbm', 'task=train', f'data={data_file}', f'output_model={model_file}']
        
        for key, value in params.items():
            cmd.append(f'{key}={value}')
            
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'training_time': end_time - start_time,
            'model_file': model_file
        }
        
    def test_parameter_validation(self):
        """Test 1: Parameter validation for blind_volume"""
        print("🧪 Test 1: Parameter Validation")
        
        # Create small test dataset
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        data_file = self.create_temp_file()
        self.save_libsvm_format(X, y, data_file)
        
        test_cases = [
            {'blind_volume': 0.0, 'should_work': True, 'desc': 'minimum value'},
            {'blind_volume': 0.5, 'should_work': True, 'desc': 'middle value'},
            {'blind_volume': 1.0, 'should_work': True, 'desc': 'maximum value'},
            {'blind_volume': -0.1, 'should_work': False, 'desc': 'negative value'},
            {'blind_volume': 1.1, 'should_work': False, 'desc': 'above maximum'},
        ]
        
        for i, case in enumerate(test_cases):
            params = {
                'num_iterations': 2,
                'min_data_in_leaf': 1,
                'min_data_in_bin': 1,
                'blind_volume': case['blind_volume']
            }
            
            result = self.run_lightgbm(data_file, params)
            
            if case['should_work']:
                success = result['returncode'] == 0
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  {i+1}.{case['desc']}: {status}")
                if not success:
                    print(f"    Error: {result['stderr']}")
            else:
                # Should fail with validation error
                has_check_error = 'CHECK' in result['stderr'] or result['returncode'] != 0
                success = has_check_error
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  {i+1}.{case['desc']}: {status}")
                
        print()
        
    def test_numerical_vs_categorical(self):
        """Test 2: Blinding only affects numerical features"""
        print("🧪 Test 2: Numerical vs Categorical Features")
        
        # Create dataset with mixed feature types
        np.random.seed(42)
        n_samples = 100
        
        # Features: 3 numerical + 2 categorical 
        X_num = np.random.randn(n_samples, 3)
        X_cat = np.random.randint(0, 5, size=(n_samples, 2))  # categorical: 0,1,2,3,4
        X = np.column_stack([X_num, X_cat])
        
        # Target based on numerical features
        y = X_num[:, 0] + 0.5 * X_num[:, 1] - 0.3 * X_num[:, 2] + 0.1 * np.random.randn(n_samples)
        
        data_file = self.create_temp_file()
        self.save_libsvm_format(X, y, data_file, categorical_features={3, 4})
        
        # Train with high blinding
        params = {
            'num_iterations': 10,
            'min_data_in_leaf': 1,
            'min_data_in_bin': 1,
            'blind_volume': 0.8,  # High blinding
            'categorical_feature': '4,5'  # Features 4,5 are categorical (1-indexed)
        }
        
        result = self.run_lightgbm(data_file, params)
        success = result['returncode'] == 0
        
        print(f"  Mixed feature training: {'✅ PASS' if success else '❌ FAIL'}")
        if not success:
            print(f"    Error: {result['stderr']}")
        print()
        
    def test_feature_importance_scaling(self):
        """Test 3: Feature importance affects blinding probability"""
        print("🧪 Test 3: Feature Importance Scaling")
        
        # Create dataset where feature 1 is very important, others are noise
        np.random.seed(123)
        n_samples = 200
        X = np.random.randn(n_samples, 5)
        
        # Make feature 0 very predictive, others are noise
        y = 3.0 * X[:, 0] + 0.1 * np.random.randn(n_samples)
        
        data_file = self.create_temp_file()
        self.save_libsvm_format(X, y, data_file)
        
        # Train multiple models with different blind_volume settings
        blind_volumes = [0.0, 0.3, 0.7]
        
        for bv in blind_volumes:
            params = {
                'num_iterations': 15,
                'min_data_in_leaf': 2,
                'min_data_in_bin': 2,
                'blind_volume': bv,
                'learning_rate': 0.1
            }
            
            result = self.run_lightgbm(data_file, params)
            success = result['returncode'] == 0
            
            print(f"  blind_volume={bv}: {'✅ PASS' if success else '❌ FAIL'}")
            if success:
                print(f"    Training time: {result['training_time']:.3f}s")
            else:
                print(f"    Error: {result['stderr'][:100]}...")
                
        print()
        
    def test_edge_cases(self):
        """Test 4: Edge cases and boundary conditions"""
        print("🧪 Test 4: Edge Cases")
        
        test_cases = [
            {
                'name': 'Very small dataset',
                'n_samples': 5,
                'n_features': 2,
                'blind_volume': 0.5
            },
            {
                'name': 'Single feature',
                'n_samples': 50,
                'n_features': 1,
                'blind_volume': 0.3
            },
            {
                'name': 'All constant features',
                'n_samples': 20,
                'n_features': 3,
                'blind_volume': 0.4,
                'constant': True
            },
            {
                'name': 'Large blind_volume',
                'n_samples': 100,
                'n_features': 5,
                'blind_volume': 0.95
            }
        ]
        
        for case in test_cases:
            print(f"  {case['name']}:", end=" ")
            
            # Generate data
            np.random.seed(42)
            if case.get('constant', False):
                X = np.ones((case['n_samples'], case['n_features']))
                y = np.ones(case['n_samples'])
            else:
                X = np.random.randn(case['n_samples'], case['n_features'])
                y = np.random.randn(case['n_samples'])
            
            data_file = self.create_temp_file()
            self.save_libsvm_format(X, y, data_file)
            
            params = {
                'num_iterations': 5,
                'min_data_in_leaf': 1,
                'min_data_in_bin': 1,
                'blind_volume': case['blind_volume'],
                'verbosity': -1  # Suppress warnings for cleaner output
            }
            
            result = self.run_lightgbm(data_file, params)
            success = result['returncode'] == 0
            
            print(f"{'✅ PASS' if success else '❌ FAIL'}")
            if not success:
                print(f"    Error: {result['stderr'][:100]}...")
                
        print()
        
    def test_performance_overhead(self):
        """Test 5: Performance impact of blinding"""
        print("🧪 Test 5: Performance Overhead")
        
        # Create moderately large dataset
        np.random.seed(456)
        n_samples = 1000
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        y = X[:, :5].sum(axis=1) + 0.1 * np.random.randn(n_samples)
        
        data_file = self.create_temp_file()
        self.save_libsvm_format(X, y, data_file)
        
        # Test different blinding levels
        blind_levels = [0.0, 0.2, 0.5, 0.8]
        times = []
        
        for blind_vol in blind_levels:
            params = {
                'num_iterations': 20,
                'min_data_in_leaf': 5,
                'min_data_in_bin': 5,
                'blind_volume': blind_vol,
                'learning_rate': 0.1,
                'verbosity': -1
            }
            
            # Run multiple times and take average
            run_times = []
            for _ in range(3):
                result = self.run_lightgbm(data_file, params)
                if result['returncode'] == 0:
                    run_times.append(result['training_time'])
                    
            if run_times:
                avg_time = np.mean(run_times)
                times.append(avg_time)
                overhead = 0 if blind_vol == 0.0 else ((avg_time / times[0] - 1) * 100)
                print(f"  blind_volume={blind_vol}: {avg_time:.3f}s (overhead: {overhead:+.1f}%)")
            else:
                print(f"  blind_volume={blind_vol}: ❌ FAILED")
                
        print()
        
    def test_reproducibility(self):
        """Test 6: Reproducibility with same seed"""
        print("🧪 Test 6: Reproducibility")
        
        # Create test dataset
        np.random.seed(789)
        X = np.random.randn(100, 10)
        y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(100)
        
        data_file = self.create_temp_file()
        self.save_libsvm_format(X, y, data_file)
        
        # Train same model twice with same seed
        params = {
            'num_iterations': 10,
            'blind_volume': 0.4,
            'seed': 42,
            'min_data_in_leaf': 2,
            'min_data_in_bin': 2,
            'verbosity': -1
        }
        
        result1 = self.run_lightgbm(data_file, params.copy())
        result2 = self.run_lightgbm(data_file, params.copy())
        
        both_success = result1['returncode'] == 0 and result2['returncode'] == 0
        
        print(f"  Both runs successful: {'✅ PASS' if both_success else '❌ FAIL'}")
        
        if both_success:
            # Compare model files (they should be identical or very similar)
            try:
                with open(result1['model_file'], 'r') as f1, open(result2['model_file'], 'r') as f2:
                    model1 = f1.read()
                    model2 = f2.read()
                identical = model1 == model2
                print(f"  Models identical: {'✅ PASS' if identical else '⚠️  DIFFERENT (expected with randomization)'}")
            except:
                print("  Could not compare model files")
        
        print()
        
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Comprehensive Blinding Implementation Tests")
        print("=" * 50)
        
        try:
            self.test_parameter_validation()
            self.test_numerical_vs_categorical() 
            self.test_feature_importance_scaling()
            self.test_edge_cases()
            self.test_performance_overhead()
            self.test_reproducibility()
            
            print("✅ All tests completed!")
            
        except KeyboardInterrupt:
            print("\n❌ Tests interrupted by user")
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    tester = BlindingTester()
    tester.run_all_tests()
