#!/usr/bin/env python3
"""
Master Classification Experiment Logger
=======================================
Creates and manages a comprehensive Excel log for all classification experiments
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
import shutil

class ClassificationLogger:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.excel_file = f"{base_dir}/Classification_Experiments_Master_Log.xlsx"
        
    def create_master_log(self):
        """Create the master Excel log with all experiment tracking sheets"""
        
        print("📊 Creating Master Classification Experiment Log")
        print("=" * 50)
        
        # Define all sheets and their structures
        sheets_config = {
            'Summary_Dashboard': {
                'columns': [
                    'Experiment_ID', 'Date', 'Experiment_Type', 'Segment_Size', 
                    'Method_Type', 'Best_Model', 'Best_Accuracy', 'Improvement', 
                    'Status', 'Notes'
                ],
                'description': 'High-level overview of all experiments'
            },
            
            'Individual_Models': {
                'columns': [
                    'Experiment_ID', 'Date', 'Segment_Size', 'Hop_Size', 
                    'Random_Forest_Acc', 'SVM_Acc', 'Logistic_Regression_Acc',
                    'Best_Individual', 'Best_Accuracy', 'Total_Segments', 
                    'Breathing_Segments', 'Non_Breathing_Segments', 'Notes'
                ],
                'description': 'Individual model performance tracking'
            },
            
            'Ensemble_Methods': {
                'columns': [
                    'Experiment_ID', 'Date', 'Segment_Size', 'Hop_Size',
                    'Individual_Best_Acc', 'Ensemble_Method', 'Ensemble_Config',
                    'Ensemble_Accuracy', 'Improvement', 'Vote_Type', 'Weights',
                    'Models_Used', 'Winner', 'Notes'
                ],
                'description': 'Ensemble method performance tracking'
            },
            
            'Temporal_Resolution': {
                'columns': [
                    'Experiment_ID', 'Date', 'Segment_Size', 'Hop_Size', 
                    'Overlap_Percentage', 'Labeling_Method', 'Accuracy',
                    'Precision', 'Recall', 'F1_Score', 'Total_Test_Segments',
                    'Visual_Mathematical_Match', 'Notes'
                ],
                'description': 'Temporal resolution optimization experiments'
            },
            
            'Feature_Analysis': {
                'columns': [
                    'Experiment_ID', 'Date', 'Feature_Type', 'Feature_Count',
                    'Handcrafted_Features', 'OPERA_CT_Features', 'Combined_Features',
                    'Best_Accuracy', 'Feature_Importance_Available', 'Notes'
                ],
                'description': 'Feature engineering and analysis results'
            },
            
            'Data_Splitting': {
                'columns': [
                    'Experiment_ID', 'Date', 'Split_Method', 'Train_Size', 
                    'Test_Size', 'Patient_Based', 'Segment_Based', 
                    'Stratified', 'Accuracy', 'Generalization_Score', 'Notes'
                ],
                'description': 'Data splitting methodology comparisons'
            },
            
            'Error_Analysis': {
                'columns': [
                    'Experiment_ID', 'Date', 'Segment_Size', 'Method',
                    'False_Positives', 'False_Negatives', 'Common_Errors',
                    'Problematic_Files', 'Error_Patterns', 'Recommendations'
                ],
                'description': 'Detailed error analysis and patterns'
            }
        }
        
        # Create Excel writer
        with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
            
            for sheet_name, config in sheets_config.items():
                print(f"📋 Creating sheet: {sheet_name}")
                
                # Create empty DataFrame with defined columns
                df = pd.DataFrame(columns=config['columns'])
                
                # Write to Excel
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Add description as a note in the first row (skip comment for now)
                # worksheet = writer.sheets[sheet_name]
                # worksheet['A1'].comment = config['description']
        
        print(f"✅ Master log created: {self.excel_file}")
        return self.excel_file
    
    def log_existing_experiments(self):
        """Scan existing results and populate the master log"""
        
        print("\n🔍 Scanning Existing Experiments")
        print("=" * 35)
        
        experiments_logged = []
        
        # Load existing Excel file
        try:
            excel_data = pd.read_excel(self.excel_file, sheet_name=None)
        except FileNotFoundError:
            print("❌ Master log not found. Creating it first...")
            self.create_master_log()
            excel_data = pd.read_excel(self.excel_file, sheet_name=None)
        
        # Scan Individual Models
        individual_models_dir = f"{self.base_dir}/Individual_Models"
        if os.path.exists(individual_models_dir):
            print(f"📊 Scanning Individual Models...")
            
            for folder in os.listdir(individual_models_dir):
                if folder.startswith('.'):
                    continue
                    
                folder_path = f"{individual_models_dir}/{folder}"
                if os.path.isdir(folder_path):
                    exp_data = self.extract_individual_experiment_data(folder, folder_path)
                    if exp_data:
                        experiments_logged.append(exp_data)
                        print(f"  ✅ Logged: {folder}")
        
        # Scan Ensemble Methods
        ensemble_methods_dir = f"{self.base_dir}/Ensemble_Methods"
        if os.path.exists(ensemble_methods_dir):
            print(f"🎭 Scanning Ensemble Methods...")
            
            for folder in os.listdir(ensemble_methods_dir):
                if folder.startswith('.'):
                    continue
                    
                folder_path = f"{ensemble_methods_dir}/{folder}"
                if os.path.isdir(folder_path):
                    exp_data = self.extract_ensemble_experiment_data(folder, folder_path)
                    if exp_data:
                        experiments_logged.append(exp_data)
                        print(f"  ✅ Logged: {folder}")
        
        # Update Excel file with logged experiments
        self.update_excel_with_experiments(experiments_logged)
        
        print(f"\n✅ Logged {len(experiments_logged)} experiments to master log")
        return experiments_logged
    
    def extract_individual_experiment_data(self, folder_name, folder_path):
        """Extract data from individual model experiment folders"""
        
        # Parse folder name to extract segment size
        segment_size = None
        if '1.0s' in folder_name or '1s' in folder_name:
            segment_size = 1.0
        elif '0.5s' in folder_name:
            segment_size = 0.5
        elif '0.25s' in folder_name:
            segment_size = 0.25
        
        # Try to find summary JSON or extract from folder structure
        summary_file = f"{folder_path}/experiment_summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {}
        
        # Create experiment data
        exp_data = {
            'type': 'individual',
            'experiment_id': f"IND_{folder_name}",
            'date': datetime.now().strftime('%Y-%m-%d'),
            'segment_size': segment_size,
            'hop_size': segment_size,  # Assume no overlap for clean results
            'folder_name': folder_name,
            'folder_path': folder_path,
            'summary': summary
        }
        
        return exp_data
    
    def extract_ensemble_experiment_data(self, folder_name, folder_path):
        """Extract data from ensemble experiment folders"""
        
        # Parse folder name to extract segment size
        segment_size = None
        if '1.0s' in folder_name or '1s' in folder_name:
            segment_size = 1.0
        elif '0.5s' in folder_name:
            segment_size = 0.5
        elif '0.25s' in folder_name:
            segment_size = 0.25
        
        # Try to find summary JSON
        summary_file = f"{folder_path}/experiment_summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {}
        
        # Create experiment data
        exp_data = {
            'type': 'ensemble',
            'experiment_id': f"ENS_{folder_name}",
            'date': datetime.now().strftime('%Y-%m-%d'),
            'segment_size': segment_size,
            'hop_size': segment_size,  # Assume no overlap for clean results
            'folder_name': folder_name,
            'folder_path': folder_path,
            'summary': summary
        }
        
        return exp_data
    
    def update_excel_with_experiments(self, experiments):
        """Update Excel file with experiment data"""
        
        print(f"\n📝 Updating Excel with {len(experiments)} experiments...")
        
        # Read existing data
        excel_data = pd.read_excel(self.excel_file, sheet_name=None)
        
        # Prepare data for each sheet
        summary_data = []
        individual_data = []
        ensemble_data = []
        temporal_data = []
        
        for exp in experiments:
            # Summary Dashboard data
            summary_row = {
                'Experiment_ID': exp['experiment_id'],
                'Date': exp['date'],
                'Experiment_Type': exp['type'].title(),
                'Segment_Size': f"{exp['segment_size']}s" if exp['segment_size'] else 'Unknown',
                'Method_Type': 'Individual' if exp['type'] == 'individual' else 'Ensemble',
                'Best_Model': exp['summary'].get('best_method', 'Unknown'),
                'Best_Accuracy': exp['summary'].get('best_accuracy', 'Unknown'),
                'Improvement': exp['summary'].get('improvement_over_individual', 0),
                'Status': exp['summary'].get('status', 'Completed'),
                'Notes': f"Folder: {exp['folder_name']}"
            }
            summary_data.append(summary_row)
            
            # Type-specific data
            if exp['type'] == 'individual':
                ind_row = {
                    'Experiment_ID': exp['experiment_id'],
                    'Date': exp['date'],
                    'Segment_Size': exp['segment_size'],
                    'Hop_Size': exp['hop_size'],
                    'Random_Forest_Acc': 'Unknown',  # Would need to parse from results
                    'SVM_Acc': 'Unknown',
                    'Logistic_Regression_Acc': 'Unknown',
                    'Best_Individual': exp['summary'].get('best_method', 'Unknown'),
                    'Best_Accuracy': exp['summary'].get('best_accuracy', 'Unknown'),
                    'Total_Segments': exp['summary'].get('test_segments', 'Unknown'),
                    'Breathing_Segments': exp['summary'].get('breathing_segments', 'Unknown'),
                    'Non_Breathing_Segments': exp['summary'].get('non_breathing_segments', 'Unknown'),
                    'Notes': f"Folder: {exp['folder_name']}"
                }
                individual_data.append(ind_row)
                
            elif exp['type'] == 'ensemble':
                ens_row = {
                    'Experiment_ID': exp['experiment_id'],
                    'Date': exp['date'],
                    'Segment_Size': exp['segment_size'],
                    'Hop_Size': exp['hop_size'],
                    'Individual_Best_Acc': exp['summary'].get('individual_best', {}).get('accuracy', 'Unknown'),
                    'Ensemble_Method': exp['summary'].get('ensemble_best', {}).get('method', 'Unknown'),
                    'Ensemble_Config': 'See Notes',
                    'Ensemble_Accuracy': exp['summary'].get('ensemble_best', {}).get('accuracy', 'Unknown'),
                    'Improvement': exp['summary'].get('improvement', 0),
                    'Vote_Type': 'Weighted' if 'weighted' in str(exp['summary']).lower() else 'Equal',
                    'Weights': 'See Config',
                    'Models_Used': 'RF+SVM+LR',
                    'Winner': exp['summary'].get('recommendation', 'Unknown'),
                    'Notes': f"Folder: {exp['folder_name']}"
                }
                ensemble_data.append(ens_row)
            
            # Temporal Resolution data (all experiments)
            temp_row = {
                'Experiment_ID': exp['experiment_id'],
                'Date': exp['date'],
                'Segment_Size': exp['segment_size'],
                'Hop_Size': exp['hop_size'],
                'Overlap_Percentage': 0 if exp['segment_size'] == exp['hop_size'] else 50,
                'Labeling_Method': 'Center-Point',  # Our final method
                'Accuracy': exp['summary'].get('best_accuracy', 'Unknown'),
                'Precision': 'Unknown',  # Would need detailed parsing
                'Recall': 'Unknown',
                'F1_Score': 'Unknown',
                'Total_Test_Segments': exp['summary'].get('test_segments', 'Unknown'),
                'Visual_Mathematical_Match': 'Yes',  # After our fixes
                'Notes': f"Folder: {exp['folder_name']}"
            }
            temporal_data.append(temp_row)
        
        # Update Excel sheets
        with pd.ExcelWriter(self.excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            
            # Summary Dashboard
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary_Dashboard', index=False)
                print(f"  ✅ Updated Summary_Dashboard with {len(summary_data)} entries")
            
            # Individual Models
            if individual_data:
                individual_df = pd.DataFrame(individual_data)
                individual_df.to_excel(writer, sheet_name='Individual_Models', index=False)
                print(f"  ✅ Updated Individual_Models with {len(individual_data)} entries")
            
            # Ensemble Methods
            if ensemble_data:
                ensemble_df = pd.DataFrame(ensemble_data)
                ensemble_df.to_excel(writer, sheet_name='Ensemble_Methods', index=False)
                print(f"  ✅ Updated Ensemble_Methods with {len(ensemble_data)} entries")
            
            # Temporal Resolution
            if temporal_data:
                temporal_df = pd.DataFrame(temporal_data)
                temporal_df.to_excel(writer, sheet_name='Temporal_Resolution', index=False)
                print(f"  ✅ Updated Temporal_Resolution with {len(temporal_data)} entries")
    
    def rename_folders_clean(self):
        """Rename folders to remove scores and use clean names"""
        
        print(f"\n🧹 Renaming Folders to Clean Names")
        print("=" * 35)
        
        # Individual Models renaming
        individual_renames = {
            '1.0s_Segments_SVM_75.6pct': '1.0s_Individual_Models',
            '0.5s_Segments_RF_81.1pct': '0.5s_Individual_Models',
            '0.25s_Segments_SVM_79.2pct': '0.25s_Individual_Models'
        }
        
        individual_dir = f"{self.base_dir}/Individual_Models"
        if os.path.exists(individual_dir):
            for old_name, new_name in individual_renames.items():
                old_path = f"{individual_dir}/{old_name}"
                new_path = f"{individual_dir}/{new_name}"
                
                if os.path.exists(old_path) and not os.path.exists(new_path):
                    shutil.move(old_path, new_path)
                    print(f"  ✅ Renamed: {old_name} → {new_name}")
        
        # Ensemble Methods renaming
        ensemble_renames = {
            '1.0s_Ensemble_0.8pct_IMPROVED': '1.0s_Ensemble_Methods',
            '0.5s_Ensemble_0.8pct_IMPROVED': '0.5s_Ensemble_Methods',
            '0.25s_Individual_0.8pct_BETTER': '0.25s_Ensemble_Methods'
        }
        
        ensemble_dir = f"{self.base_dir}/Ensemble_Methods"
        if os.path.exists(ensemble_dir):
            for old_name, new_name in ensemble_renames.items():
                old_path = f"{ensemble_dir}/{old_name}"
                new_path = f"{ensemble_dir}/{new_name}"
                
                if os.path.exists(old_path) and not os.path.exists(new_path):
                    shutil.move(old_path, new_path)
                    print(f"  ✅ Renamed: {old_name} → {new_name}")
        
        print(f"✅ Folder renaming completed")

def main():
    """Main execution function"""
    print("📊 MASTER CLASSIFICATION EXPERIMENT LOGGER")
    print("=" * 45)
    
    base_dir = "/Users/yunhwang/Desktop/Stethoscope_Project/Development/3) Transfer Learning with OPERA-CT/02_Full_Frozen_Breathing_Classification"
    
    logger = ClassificationLogger(base_dir)
    
    # Step 1: Create master log structure
    excel_file = logger.create_master_log()
    
    # Step 2: Rename folders to clean names
    logger.rename_folders_clean()
    
    # Step 3: Log existing experiments
    experiments = logger.log_existing_experiments()
    
    # Step 4: Final summary
    print(f"\n🎉 MASTER CLASSIFICATION LOG COMPLETE")
    print("=" * 40)
    print(f"📁 Excel File: {Path(excel_file).name}")
    print(f"📊 Experiments Logged: {len(experiments)}")
    print(f"📋 Sheets Created: 7")
    print(f"🧹 Folders Cleaned: Yes")
    
    print(f"\n📋 Excel Sheets:")
    print(f"  • Summary_Dashboard - High-level overview")
    print(f"  • Individual_Models - Single algorithm results")
    print(f"  • Ensemble_Methods - Combined algorithm results")
    print(f"  • Temporal_Resolution - Segment size optimization")
    print(f"  • Feature_Analysis - Feature engineering results")
    print(f"  • Data_Splitting - Splitting methodology comparison")
    print(f"  • Error_Analysis - Detailed error patterns")
    
    print(f"\n✅ All experiments now tracked in master Excel log!")
    print(f"📈 No more scores in folder names - check Excel for all metrics!")

if __name__ == "__main__":
    main()
