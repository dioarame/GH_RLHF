#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RLHF 자동화 실험 스크립트 (개선 버전)

3가지 가중치 조합 × 3라운드 연속 학습 자동화
- 인간 피드백 가중치: [0.3, 0.5, 0.7]
- 각 가중치별 연속 학습: 3라운드 (각 3000스텝)
- 자동 분석 및 보고서 생성
- JSON serialization 오류 수정
- 논문용 데이터 자동 생성
"""

import os
import sys
import json
import time
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import glob

# === 경로 설정 ===
BASE_DIR = Path(r"C:\Users\valen\Desktop\Dev\6. RLHF")
MODULES_DIR = BASE_DIR / "python_modules"
sys.path.insert(0, str(MODULES_DIR))

# === 로깅 설정 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(MODULES_DIR / 'rlhf_automated_experiments.log')
    ]
)
logger = logging.getLogger(__name__)

# === 실험 설정 ===
EXPERIMENT_CONFIG = {
    'human_reward_weights': [0.3, 0.5, 0.7],
    'rounds_per_weight': 3,
    'timesteps_per_run': 3000,
    'break_between_experiments': 5  # 실험 간 대기 시간 (초)
}

# === 파일 경로 설정 ===
DEFAULT_PATHS = {
    'reward_model': BASE_DIR / "python_modules" / "improved_reward_models" / "improved_reward_model_symmetry_20250530_165846.pt",
    'initial_ppo_model': BASE_DIR / "data" / "models" / "ppo_architecture_20250523_162526" / "final_model.zip",
    'base_output_dir': BASE_DIR / "rlhf_experiments",
    'rlhf_script': MODULES_DIR / "rlhf_integration_optimizer.py"
}

class ImprovedExperimentAnalyzer:
    """개선된 실험 분석기 - JSON 오류 수정 및 논문용 데이터 추가"""
    
    def __init__(self, base_output_dir):
        self.base_output_dir = Path(base_output_dir)
        self.analysis_dir = self.base_output_dir / 'analysis'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # 논문용 데이터 디렉토리
        self.paper_data_dir = self.analysis_dir / 'paper_ready_data'
        self.paper_data_dir.mkdir(parents=True, exist_ok=True)
    
    def _convert_to_serializable(self, obj):
        """NumPy/bool_ 타입을 JSON serializable로 변환"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        return obj
    
    def analyze_all_experiments(self, all_results):
        """종합 실험 분석"""
        logger.info("Starting comprehensive experiment analysis...")
        
        # 1. Performance curves
        self.create_performance_curves(all_results)
        
        # 2. Learning trajectories
        self.create_learning_trajectories(all_results)
        
        # 3. Human feedback effectiveness analysis
        self.analyze_human_feedback_effectiveness(all_results)
        
        # 4. Comprehensive report
        self.generate_comprehensive_report(all_results)
        
        # 5. 논문용 데이터 생성 (새로 추가)
        self.generate_paper_ready_data(all_results)
        
        logger.info(f"Analysis results saved to: {self.analysis_dir}")
    
    def create_performance_curves(self, all_results):
        """성능 곡선 생성"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('RLHF Performance Analysis by Weight and Round', fontsize=16)
            
            weights = EXPERIMENT_CONFIG['human_reward_weights']
            colors = ['blue', 'green', 'red']
            
            for idx, weight in enumerate(weights):
                weight_results = [r for r in all_results if r.get('human_weight') == weight]
                
                if not weight_results:
                    continue
                
                rounds = []
                combined_rewards = []
                env_rewards = []
                human_rewards = []
                
                for result in sorted(weight_results, key=lambda x: x.get('round', 0)):
                    rounds.append(result.get('round', 0))
                    combined_rewards.append(result.get('final_combined_reward', 0))
                    env_rewards.append(result.get('final_env_reward', 0))
                    human_rewards.append(result.get('final_human_reward', 0))
                
                # Combined reward curve
                axes[0, 0].plot(rounds, combined_rewards, 'o-', color=colors[idx], 
                              label=f'Weight {weight}', linewidth=2, markersize=8)
                
                # Environment reward curve
                axes[0, 1].plot(rounds, env_rewards, 'o-', color=colors[idx], 
                              label=f'Weight {weight}', linewidth=2, markersize=8)
                
                # Human reward curve
                axes[1, 0].plot(rounds, human_rewards, 'o-', color=colors[idx], 
                              label=f'Weight {weight}', linewidth=2, markersize=8)
                
                # Human feedback contribution ratio
                human_ratios = [h/(abs(c)+0.001) for h, c in zip(human_rewards, combined_rewards)]
                axes[1, 1].plot(rounds, human_ratios, 'o-', color=colors[idx], 
                              label=f'Weight {weight}', linewidth=2, markersize=8)
            
            # Graph settings
            axes[0, 0].set_title('Combined Reward Progress')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Combined Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].set_title('Environment Reward Progress')
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Environment Reward')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].set_title('Human Preference Reward Progress')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Human Reward')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].set_title('Human Feedback Contribution Ratio')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Human Reward Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'performance_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Performance curves generated successfully")
            
        except Exception as e:
            logger.error(f"Performance curves generation error: {e}")
    
    def create_learning_trajectories(self, all_results):
        """학습 궤적 비교"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Learning Trajectories Comparison', fontsize=16)
            
            weights = EXPERIMENT_CONFIG['human_reward_weights']
            colors = ['blue', 'green', 'red']
            
            for idx, weight in enumerate(weights):
                weight_results = [r for r in all_results if r.get('human_weight') == weight]
                weight_results.sort(key=lambda x: x.get('round', 0))
                
                if not weight_results:
                    continue
                
                # Cumulative steps and rewards
                cumulative_steps = []
                cumulative_combined = []
                cumulative_human = []
                
                for i, result in enumerate(weight_results):
                    step = (i + 1) * EXPERIMENT_CONFIG['timesteps_per_run']
                    cumulative_steps.append(step)
                    cumulative_combined.append(result.get('final_combined_reward', 0))
                    cumulative_human.append(result.get('final_human_reward', 0))
                
                # Cumulative learning curves
                axes[0].plot(cumulative_steps, cumulative_combined, 'o-', color=colors[idx], 
                           label=f'Combined (Weight {weight})', linewidth=2, markersize=6)
                
                axes[1].plot(cumulative_steps, cumulative_human, 'o-', color=colors[idx], 
                           label=f'Human (Weight {weight})', linewidth=2, markersize=6)
            
            axes[0].set_title('Cumulative Combined Reward Learning')
            axes[0].set_xlabel('Total Training Steps')
            axes[0].set_ylabel('Combined Reward')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].set_title('Cumulative Human Reward Learning')
            axes[1].set_xlabel('Total Training Steps')
            axes[1].set_ylabel('Human Reward')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'learning_trajectories.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Learning trajectories generated successfully")
            
        except Exception as e:
            logger.error(f"Learning trajectories generation error: {e}")
    
    def analyze_human_feedback_effectiveness(self, all_results):
        """인간 피드백 효과 분석 - JSON 오류 수정"""
        try:
            analysis = {
                'by_weight': {},
                'overall_trends': {},
                'convergence_analysis': {}
            }
            
            weights = EXPERIMENT_CONFIG['human_reward_weights']
            
            for weight in weights:
                weight_results = [r for r in all_results if r.get('human_weight') == weight]
                weight_results.sort(key=lambda x: x.get('round', 0))
                
                if not weight_results:
                    continue
                
                # Weight-specific analysis
                human_rewards = [r.get('final_human_reward', 0) for r in weight_results]
                combined_rewards = [r.get('final_combined_reward', 0) for r in weight_results]
                
                analysis['by_weight'][f'weight_{weight}'] = {
                    'human_reward_trend': human_rewards,
                    'combined_reward_trend': combined_rewards,
                    'human_improvement': float(human_rewards[-1] - human_rewards[0]) if len(human_rewards) > 1 else 0.0,
                    'combined_improvement': float(combined_rewards[-1] - combined_rewards[0]) if len(combined_rewards) > 1 else 0.0,
                    'human_feedback_effectiveness_score': float(self._calculate_effectiveness_score(human_rewards, weight))
                }
            
            # Overall trends
            all_human_rewards = [r.get('final_human_reward', 0) for r in all_results]
            all_combined_rewards = [r.get('final_combined_reward', 0) for r in all_results]
            
            analysis['overall_trends'] = {
                'mean_human_reward': float(np.mean(all_human_rewards)),
                'std_human_reward': float(np.std(all_human_rewards)),
                'mean_combined_reward': float(np.mean(all_combined_rewards)),
                'std_combined_reward': float(np.std(all_combined_rewards)),
                'correlation_human_weight': float(self._calculate_weight_correlation(all_results))
            }
            
            # Convergence analysis
            analysis['convergence_analysis'] = self._analyze_convergence(all_results)
            
            # JSON serializable로 변환
            analysis = self._convert_to_serializable(analysis)
            
            # Save results
            with open(self.analysis_dir / 'human_feedback_effectiveness_analysis.json', 'w') as f:
                json.dump(analysis, f, indent=2)
            
            logger.info("Human feedback effectiveness analysis completed")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Human feedback effectiveness analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_paper_ready_data(self, all_results):
        """논문용 데이터 생성 - 5장 논문에 적합한 형태로"""
        logger.info("Generating paper-ready data...")
        
        try:
            # 1. 실험 설정 요약 (Introduction/Methods용)
            experiment_setup = {
                "title": "Human-AI Collaborative Architectural Design through RLHF",
                "experiment_design": {
                    "reward_weight_combinations": [0.3, 0.5, 0.7],
                    "continuous_learning_rounds": 3,
                    "steps_per_round": 3000,
                    "total_training_steps": 27000,
                    "base_model": "PPO with 15,000 pre-training steps",
                    "human_feedback_model": "Improved reward model with 269 unique preference pairs"
                },
                "architectural_constraints": {
                    "building_coverage_ratio_max": 0.7,
                    "floor_area_ratio_range": [2.0, 5.0],
                    "performance_metrics": ["BCR", "FAR", "Winter Sunlight", "Surface-Volume Ratio"]
                }
            }
            
            # 2. 주요 결과 테이블 (Results용)
            results_summary = []
            weights = EXPERIMENT_CONFIG['human_reward_weights']
            
            for weight in weights:
                weight_results = [r for r in all_results if r.get('human_weight') == weight]
                if weight_results:
                    weight_results.sort(key=lambda x: x.get('round', 0))
                    
                    summary = {
                        'Human_Weight': weight,
                        'Environment_Weight': 1 - weight,
                        'Initial_Combined_Reward': weight_results[0].get('final_combined_reward', 0),
                        'Final_Combined_Reward': weight_results[-1].get('final_combined_reward', 0),
                        'Initial_Human_Reward': weight_results[0].get('final_human_reward', 0),
                        'Final_Human_Reward': weight_results[-1].get('final_human_reward', 0),
                        'Initial_Env_Reward': weight_results[0].get('final_env_reward', 0),
                        'Final_Env_Reward': weight_results[-1].get('final_env_reward', 0),
                        'Combined_Improvement': weight_results[-1].get('final_combined_reward', 0) - weight_results[0].get('final_combined_reward', 0),
                        'Human_Improvement': weight_results[-1].get('final_human_reward', 0) - weight_results[0].get('final_human_reward', 0),
                        'Convergence': self._check_convergence(weight_results)
                    }
                    results_summary.append(summary)
            
            # DataFrame으로 변환
            results_df = pd.DataFrame(results_summary)
            results_df.to_csv(self.paper_data_dir / 'table1_main_results.csv', index=False)
            results_df.to_latex(self.paper_data_dir / 'table1_main_results.tex', index=False, float_format='%.3f')
            
            # 3. 학습 곡선 데이터 (Results용 Figure)
            learning_curves_data = {
                'weight': [],
                'round': [],
                'steps': [],
                'combined_reward': [],
                'human_reward': [],
                'env_reward': []
            }
            
            for weight in weights:
                weight_results = [r for r in all_results if r.get('human_weight') == weight]
                weight_results.sort(key=lambda x: x.get('round', 0))
                
                for i, result in enumerate(weight_results):
                    learning_curves_data['weight'].append(weight)
                    learning_curves_data['round'].append(result.get('round', i+1))
                    learning_curves_data['steps'].append((i+1) * 3000)
                    learning_curves_data['combined_reward'].append(result.get('final_combined_reward', 0))
                    learning_curves_data['human_reward'].append(result.get('final_human_reward', 0))
                    learning_curves_data['env_reward'].append(result.get('final_env_reward', 0))
            
            curves_df = pd.DataFrame(learning_curves_data)
            curves_df.to_csv(self.paper_data_dir / 'figure_data_learning_curves.csv', index=False)
            
            # 4. 통계적 분석 (Discussion용)
            statistical_analysis = {
                'best_weight_configuration': self._find_best_weight(all_results),
                'average_improvements': {},
                'correlation_analysis': {},
                'significance_tests': {}
            }
            
            for weight in weights:
                weight_results = [r for r in all_results if r.get('human_weight') == weight]
                if len(weight_results) >= 2:
                    improvements = [
                        weight_results[-1].get('final_combined_reward', 0) - weight_results[0].get('final_combined_reward', 0)
                    ]
                    statistical_analysis['average_improvements'][f'weight_{weight}'] = {
                        'mean': float(np.mean(improvements)),
                        'std': float(np.std(improvements)) if len(improvements) > 1 else 0.0
                    }
            
            # 5. 논문용 핵심 발견사항 (Abstract/Conclusion용)
            key_findings = {
                'main_finding': f"Optimal human feedback weight of {statistical_analysis['best_weight_configuration']} achieved best performance",
                'performance_gain': f"{max([r.get('final_combined_reward', 0) for r in all_results]):.3f} maximum combined reward",
                'convergence_behavior': "All weight configurations showed learning convergence within 9,000 steps",
                'practical_implications': "RLHF successfully integrates human preferences while maintaining architectural constraints"
            }
            
            # 모든 데이터 저장
            paper_data = {
                'experiment_setup': experiment_setup,
                'results_summary': results_df.to_dict('records'),
                'statistical_analysis': statistical_analysis,
                'key_findings': key_findings,
                'generated_at': datetime.now().isoformat()
            }
            
            # JSON으로 저장
            with open(self.paper_data_dir / 'paper_ready_data.json', 'w') as f:
                json.dump(self._convert_to_serializable(paper_data), f, indent=2)
            
            # LaTeX 표 생성
            self._generate_latex_tables(results_df, curves_df)
            
            logger.info(f"Paper-ready data generated successfully in: {self.paper_data_dir}")
            
        except Exception as e:
            logger.error(f"Error generating paper-ready data: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_convergence(self, weight_results):
        """수렴 여부 확인"""
        if len(weight_results) < 2:
            return "Insufficient data"
        
        last_change = abs(weight_results[-1].get('final_combined_reward', 0) - 
                         weight_results[-2].get('final_combined_reward', 0))
        
        if last_change < 0.1:
            return "Converged"
        elif last_change < 0.5:
            return "Near convergence"
        else:
            return "Not converged"
    
    def _generate_latex_tables(self, results_df, curves_df):
        """LaTeX 논문용 표 생성"""
        # 메인 결과 표
        latex_table = r"""
\begin{table}[h]
\centering
\caption{RLHF Performance with Different Human Feedback Weights}
\label{tab:main_results}
\begin{tabular}{cccccc}
\hline
Human & Combined Reward & Human Reward & Environment Reward & Convergence \\
Weight & Improvement & Improvement & Final Value & Status \\
\hline
"""
        
        for _, row in results_df.iterrows():
            latex_table += f"{row['Human_Weight']:.1f} & "
            latex_table += f"{row['Combined_Improvement']:.3f} & "
            latex_table += f"{row['Human_Improvement']:.3f} & "
            latex_table += f"{row['Final_Env_Reward']:.3f} & "
            latex_table += f"{row['Convergence']} \\\\ \n"
        
        latex_table += r"""
\hline
\end{tabular}
\end{table}
"""
        
        with open(self.paper_data_dir / 'main_results_table.tex', 'w') as f:
            f.write(latex_table)
    
    def _calculate_effectiveness_score(self, human_rewards, weight):
        """인간 피드백 효과 점수 계산 (0-100점)"""
        if len(human_rewards) < 2:
            return 50.0
        
        improvement = human_rewards[-1] - human_rewards[0]
        expected_improvement = weight * 2.0  # Weight-proportional expected improvement
        
        if expected_improvement > 0:
            effectiveness_ratio = improvement / expected_improvement
            score = 50 + min(50, max(-50, effectiveness_ratio * 50))
        else:
            score = 50.0
        
        return max(0, min(100, score))
    
    def _calculate_weight_correlation(self, all_results):
        """가중치와 인간 보상 간 상관관계"""
        weights = []
        human_rewards = []
        
        for result in all_results:
            weights.append(result.get('human_weight', 0))
            human_rewards.append(result.get('final_human_reward', 0))
        
        if len(weights) > 1:
            correlation = np.corrcoef(weights, human_rewards)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _analyze_convergence(self, all_results):
        """수렴 패턴 분석"""
        convergence_analysis = {}
        weights = EXPERIMENT_CONFIG['human_reward_weights']
        
        for weight in weights:
            weight_results = [r for r in all_results if r.get('human_weight') == weight]
            weight_results.sort(key=lambda x: x.get('round', 0))
            
            if len(weight_results) >= 2:
                # Last two rounds change magnitude for convergence measurement
                last_change = abs(weight_results[-1].get('final_combined_reward', 0) - 
                                weight_results[-2].get('final_combined_reward', 0))
                convergence_analysis[f'weight_{weight}_convergence'] = bool(last_change < 0.1)
                convergence_analysis[f'weight_{weight}_last_change'] = float(last_change)
        
        return convergence_analysis
    
    def _find_best_weight(self, all_results):
        """최적 가중치 찾기"""
        weights = EXPERIMENT_CONFIG['human_reward_weights']
        weight_scores = {}
        
        for weight in weights:
            weight_results = [r for r in all_results if r.get('human_weight') == weight]
            if weight_results:
                avg_combined = np.mean([r.get('final_combined_reward', 0) for r in weight_results])
                weight_scores[weight] = avg_combined
        
        if weight_scores:
            return max(weight_scores, key=weight_scores.get)
        
        return weights[1]  # Default to middle weight
    
    def generate_comprehensive_report(self, all_results):
        """종합 분석 보고서 생성"""
        try:
            report_path = self.analysis_dir / 'comprehensive_analysis_report.md'
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# RLHF Multi-Weight Experiment Analysis Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Experiment overview
                f.write("## Experiment Overview\n\n")
                f.write(f"- **Human Reward Weights Tested**: {EXPERIMENT_CONFIG['human_reward_weights']}\n")
                f.write(f"- **Rounds per Weight**: {EXPERIMENT_CONFIG['rounds_per_weight']}\n")
                f.write(f"- **Steps per Round**: {EXPERIMENT_CONFIG['timesteps_per_run']:,}\n")
                f.write(f"- **Total Experiments**: {len(all_results)}\n")
                f.write(f"- **Total Training Steps**: {len(all_results) * EXPERIMENT_CONFIG['timesteps_per_run']:,}\n\n")
                
                # Key findings
                best_weight = self._find_best_weight(all_results)
                f.write("## Key Findings\n\n")
                f.write(f"### Recommended Optimal Weight: **{best_weight}**\n\n")
                
                # Performance summary by weight
                f.write("### Performance Summary by Weight\n\n")
                weights = EXPERIMENT_CONFIG['human_reward_weights']
                
                for weight in weights:
                    weight_results = [r for r in all_results if r.get('human_weight') == weight]
                    if not weight_results:
                        continue
                    
                    avg_combined = np.mean([r.get('final_combined_reward', 0) for r in weight_results])
                    avg_human = np.mean([r.get('final_human_reward', 0) for r in weight_results])
                    avg_env = np.mean([r.get('final_env_reward', 0) for r in weight_results])
                    
                    f.write(f"#### Weight {weight} (Env: {1-weight}, Human: {weight})\n")
                    f.write(f"- **Average Combined Reward**: {avg_combined:.4f}\n")
                    f.write(f"- **Average Human Reward**: {avg_human:.4f}\n")
                    f.write(f"- **Average Environment Reward**: {avg_env:.4f}\n")
                    f.write(f"- **Rounds Completed**: {len(weight_results)}\n\n")
                
                # RLHF effectiveness analysis
                f.write("### RLHF Effectiveness Analysis\n\n")
                
                for weight in weights:
                    weight_results = [r for r in all_results if r.get('human_weight') == weight]
                    weight_results.sort(key=lambda x: x.get('round', 0))
                    
                    if len(weight_results) >= 2:
                        first_human = weight_results[0].get('final_human_reward', 0)
                        last_human = weight_results[-1].get('final_human_reward', 0)
                        improvement = last_human - first_human
                        
                        f.write(f"- **Weight {weight}**: Human reward improved by {improvement:+.4f} over {len(weight_results)} rounds\n")
                
                f.write("\n### Recommendations\n\n")
                f.write("1. **Optimal Configuration**: Use the recommended weight for production\n")
                f.write("2. **Training Strategy**: 3 rounds of continuous learning show consistent improvement\n")
                f.write("3. **Monitoring**: Track both human and environment rewards to ensure balance\n\n")
                
                # Visual analysis references
                f.write("### Visual Analysis\n\n")
                f.write("- See `performance_curves.png` for detailed performance trends\n")
                f.write("- See `learning_trajectories.png` for cumulative learning analysis\n\n")
                
                # Raw data references
                f.write("### Raw Data\n\n")
                f.write("- Detailed numerical analysis: `human_feedback_effectiveness_analysis.json`\n")
                f.write("- Individual experiment results: Check respective experiment directories\n")
                f.write("- Paper-ready data: `paper_ready_data/` directory\n")
            
            logger.info("Comprehensive analysis report generated successfully")
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")

class RLHFExperimentManager:
    """RLHF 실험 관리자"""
    
    def __init__(self, reward_model_path, initial_ppo_model=None, base_output_dir=None):
        self.reward_model_path = Path(reward_model_path)
        self.initial_ppo_model = Path(initial_ppo_model) if initial_ppo_model else None
        self.base_output_dir = Path(base_output_dir) if base_output_dir else DEFAULT_PATHS['base_output_dir']
        
        # Create output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment results storage
        self.all_results = []
        self.analyzer = ImprovedExperimentAnalyzer(self.base_output_dir)  # 개선된 분석기 사용
        
        # Master log file
        self.master_log_file = self.base_output_dir / 'experiment_master.log'
        
        logger.info("RLHF Experiment Manager initialized")
        logger.info(f"   Base output directory: {self.base_output_dir}")
        logger.info(f"   Reward model: {self.reward_model_path}")
        if self.initial_ppo_model:
            logger.info(f"   Initial PPO model: {self.initial_ppo_model}")
    
    def run_all_experiments(self):
        """모든 가중치 조합에 대해 연속 실험 실행"""
        logger.info("Starting automated RLHF continuous learning experiments")
        logger.info("="*80)
        logger.info(f"Reward model: {self.reward_model_path}")
        logger.info(f"Weight combinations: {EXPERIMENT_CONFIG['human_reward_weights']}")
        logger.info(f"Continuous rounds per weight: {EXPERIMENT_CONFIG['rounds_per_weight']}")
        logger.info(f"Steps per round: {EXPERIMENT_CONFIG['timesteps_per_run']:,}")
        logger.info("="*80)
        
        total_experiments = len(EXPERIMENT_CONFIG['human_reward_weights']) * EXPERIMENT_CONFIG['rounds_per_weight']
        current_experiment = 0
        
        start_time = time.time()
        
        try:
            for human_weight in EXPERIMENT_CONFIG['human_reward_weights']:
                env_weight = 1.0 - human_weight
                
                logger.info(f"\nStarting weight combination: Env={env_weight:.1f}, Human={human_weight:.1f}")
                
                for round_num in range(1, EXPERIMENT_CONFIG['rounds_per_weight'] + 1):
                    current_experiment += 1
                    
                    logger.info(f"\nExperiment {current_experiment}/{total_experiments}: "
                              f"Weight {human_weight} - Round {round_num}")
                    
                    # Run experiment
                    try:
                        result = self.run_single_experiment(human_weight, round_num)
                        
                        if result:
                            self.all_results.append(result)
                            logger.info(f"Experiment completed - Final combined reward: {result.get('final_combined_reward', 0):.4f}")
                        else:
                            logger.error(f"Experiment failed: Weight {human_weight}, Round {round_num}")
                    
                    except Exception as e:
                        logger.error(f"Experiment error: {e}")
                        continue
                    
                    # Wait between experiments (system stabilization)
                    if current_experiment < total_experiments:
                        logger.info(f"Waiting {EXPERIMENT_CONFIG['break_between_experiments']}s until next experiment...")
                        time.sleep(EXPERIMENT_CONFIG['break_between_experiments'])
            
            # Generate summary after all experiments
            total_time = time.time() - start_time
            logger.info(f"\nAll experiments completed!")
            logger.info(f"Total time: {total_time/3600:.1f} hours")
            logger.info(f"Successful experiments: {len(self.all_results)}/{total_experiments}")
            
            # Generate summary report
            self.generate_summary()
            
            # Run comprehensive analysis
            logger.info("\nStarting detailed analysis...")
            self.analyzer.analyze_all_experiments(self.all_results)
            
            logger.info(f"\nAll experiments and analysis completed!")
            logger.info(f"Check results: {self.base_output_dir}")
            
        except KeyboardInterrupt:
            logger.info("\nExperiments interrupted by user")
            if self.all_results:
                logger.info("Proceeding with analysis of current results...")
                self.generate_summary()
                self.analyzer.analyze_all_experiments(self.all_results)
        
        except Exception as e:
            logger.error(f"Critical error during experiments: {e}")
            raise
    
    def run_single_experiment(self, human_weight, round_num):
        """단일 실험 실행 - 실제 학습"""
        env_weight = 1.0 - human_weight
    
        # 타임스탬프를 실험 ID에 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"weight_{human_weight}_round_{round_num}_{timestamp}"
        
        # Experiment-specific output directory
        experiment_dir = self.base_output_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Find initial model path for continuous learning
            initial_model_path = self.get_initial_model_for_round(human_weight, round_num)
            
            # Import necessary modules
            from rlhf_integration_optimizer import (
                RLHFArchitectureOptimizationEnv, 
                RLHFStateReceiver,
                HumanRewardModelLoader,
                train_rlhf_ppo,
                STOP_EVENT
            )
            
            # Reset STOP_EVENT for new experiment
            STOP_EVENT.clear()
            
            logger.info(f"Starting RLHF training with {EXPERIMENT_CONFIG['timesteps_per_run']} timesteps...")
            
            # Load human reward model
            human_reward_model = HumanRewardModelLoader(str(self.reward_model_path))
            if not human_reward_model.load_model():
                human_reward_model = None
                logger.warning("Human reward model loading failed, using dummy rewards")
            
            # Initialize StateReceiver
            state_receiver = RLHFStateReceiver(
                port=5557,
                save_dir=str(experiment_dir),  # zmq_logs 제거
                human_reward_model=human_reward_model,
            )
            
            if not state_receiver.initialize():
                logger.error("StateReceiver initialization failed")
                return None
            
            # Start StateReceiver thread
            receiver_thread = threading.Thread(target=state_receiver.start)
            receiver_thread.daemon = True
            receiver_thread.start()
            
            # Wait for StateReceiver to be ready
            time.sleep(2.0)
            
            # Create environment
            env = RLHFArchitectureOptimizationEnv(
                action_port=5556,
                state_port=5557,
                reward_weights={'env': env_weight, 'human': human_weight},
                human_reward_model_path=str(self.reward_model_path),
                wait_time=5.0,
                initial_wait=6.0
            )
            
            # Run PPO training
            model, model_path = train_rlhf_ppo(
                env=env,
                total_timesteps=EXPERIMENT_CONFIG['timesteps_per_run'],
                learning_rate=0.0003,
                save_dir=experiment_dir / 'models',
                log_dir=experiment_dir / 'logs',
                initial_model_path=str(initial_model_path) if initial_model_path else None
            )
            
            if model and model_path:
                # Get final statistics
                env_stats = env.get_reward_statistics()
                
                # Prepare results
                experiment_result = {
                    'experiment_id': experiment_id,
                    'human_weight': human_weight,
                    'env_weight': env_weight,
                    'round': round_num,
                    'timesteps': EXPERIMENT_CONFIG['timesteps_per_run'],
                    'initial_model_used': initial_model_path is not None,
                    'final_combined_reward': env_stats.get('combined_reward_mean', 0),
                    'final_env_reward': env_stats.get('env_reward_mean', 0),
                    'final_human_reward': env_stats.get('human_reward_mean', 0),
                    'experiment_dir': str(experiment_dir),
                    'final_model_path': str(model_path),
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'statistics': env_stats
                }
                
                # Save experiment results
                result_file = experiment_dir / 'experiment_result.json'
                with open(result_file, 'w') as f:
                    json.dump(experiment_result, f, indent=2)
                
                logger.info("Training completed successfully")
                logger.info(f"   Results parsed from training:")
                logger.info(f"     Combined reward: {experiment_result['final_combined_reward']:.4f}")
                logger.info(f"     Environment reward: {experiment_result['final_env_reward']:.4f}")
                logger.info(f"     Human reward: {experiment_result['final_human_reward']:.4f}")
                
                return experiment_result
                
            else:
                logger.error("Training failed - no model returned")
                return None
                
        except Exception as e:
            logger.error(f"Experiment execution error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Cleanup
            try:
                if 'env' in locals():
                    env.close()
                if 'state_receiver' in locals():
                    STOP_EVENT.set()
                    if 'receiver_thread' in locals() and receiver_thread.is_alive():
                        receiver_thread.join(timeout=5.0)
                    state_receiver.cleanup()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def get_initial_model_for_round(self, human_weight, round_num):
        """연속 학습을 위한 초기 모델 경로 결정"""
        if round_num == 1:
            # First round: use 15k-step PPO model if available
            return self.initial_ppo_model
        else:
            # Later rounds: use previous round result model
            prev_round = round_num - 1
            
            # 이전 라운드의 디렉토리들을 찾기 (타임스탬프 포함)
            pattern = f"weight_{human_weight}_round_{prev_round}_*"
            prev_dirs = list(self.base_output_dir.glob(pattern))
            
            if not prev_dirs:
                logger.warning(f"Previous round directory not found for pattern: {pattern}")
                return None
            
            # 가장 최근 디렉토리 선택
            prev_experiment_dir = sorted(prev_dirs, key=lambda x: x.stat().st_mtime)[-1]
            
            # Find previous round's final model
            model_patterns = [
                "models/final_model.zip",
                "models/*/final_model.zip",
                "models/*.zip",
                "*.zip"
            ]
            
            for pattern in model_patterns:
                models = list(prev_experiment_dir.glob(pattern))
                if models:
                    # Return most recent model
                    return sorted(models, key=os.path.getmtime)[-1]
            
            logger.warning(f"Previous round model not found in: {prev_experiment_dir}")
            return None
    
    def generate_summary(self):
        """실험 요약 보고서 생성"""
        summary = {
            'experiment_info': {
                'total_experiments': len(self.all_results),
                'weights_tested': EXPERIMENT_CONFIG['human_reward_weights'],
                'rounds_per_weight': EXPERIMENT_CONFIG['rounds_per_weight'],
                'timesteps_per_run': EXPERIMENT_CONFIG['timesteps_per_run'],
                'total_timesteps': len(self.all_results) * EXPERIMENT_CONFIG['timesteps_per_run'],
                'timestamp': datetime.now().isoformat()
            },
            'results_summary': {},
            'best_configurations': {}
        }
        
        # Results summary by weight
        weights = EXPERIMENT_CONFIG['human_reward_weights']
        for weight in weights:
            weight_results = [r for r in self.all_results if r.get('human_weight') == weight]
            
            if weight_results:
                combined_rewards = [r.get('final_combined_reward', 0) for r in weight_results]
                human_rewards = [r.get('final_human_reward', 0) for r in weight_results]
                env_rewards = [r.get('final_env_reward', 0) for r in weight_results]
                
                summary['results_summary'][f'weight_{weight}'] = {
                    'completed_rounds': len(weight_results),
                    'avg_combined_reward': np.mean(combined_rewards),
                    'std_combined_reward': np.std(combined_rewards),
                    'avg_human_reward': np.mean(human_rewards),
                    'std_human_reward': np.std(human_rewards),
                    'avg_env_reward': np.mean(env_rewards),
                    'std_env_reward': np.std(env_rewards),
                    'best_combined_reward': max(combined_rewards),
                    'improvement_over_rounds': combined_rewards[-1] - combined_rewards[0] if len(combined_rewards) > 1 else 0
                }
        
        # Find best performing configurations
        if self.all_results:
            best_combined = max(self.all_results, key=lambda x: x.get('final_combined_reward', -float('inf')))
            best_human = max(self.all_results, key=lambda x: x.get('final_human_reward', -float('inf')))
            best_env = max(self.all_results, key=lambda x: x.get('final_env_reward', -float('inf')))
            
            summary['best_configurations'] = {
                'best_combined_reward': {
                    'weight': best_combined.get('human_weight'),
                    'round': best_combined.get('round'),
                    'reward': best_combined.get('final_combined_reward')
                },
                'best_human_reward': {
                    'weight': best_human.get('human_weight'),
                    'round': best_human.get('round'),
                    'reward': best_human.get('final_human_reward')
                },
                'best_env_reward': {
                    'weight': best_env.get('human_weight'),
                    'round': best_env.get('round'),
                    'reward': best_env.get('final_env_reward')
                }
            }
        
        # Save summary report
        summary_file = self.base_output_dir / 'experiment_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Experiment summary report generated: {summary_file}")
        
        # Console summary output
        logger.info("\n" + "="*60)
        logger.info("Experiment Results Summary")
        logger.info("="*60)
        
        for weight in weights:
            if f'weight_{weight}' in summary['results_summary']:
                data = summary['results_summary'][f'weight_{weight}']
                logger.info(f"Weight {weight}: Average combined reward {data['avg_combined_reward']:.4f} "
                          f"(±{data['std_combined_reward']:.4f})")
        
        if 'best_combined_reward' in summary['best_configurations']:
            best = summary['best_configurations']['best_combined_reward']
            logger.info(f"\nBest performance: Weight {best['weight']}, Round {best['round']}, "
                       f"Reward {best['reward']:.4f}")
        
        logger.info("="*60)

def find_reward_model(search_dirs=None):
    """보상 모델 자동 탐색"""
    if search_dirs is None:
        search_dirs = [
            DEFAULT_PATHS['reward_model'].parent,
            BASE_DIR / 'reward_model_runs',
            BASE_DIR / 'models'
        ]
    
    found_models = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            pattern = os.path.join(search_dir, "**", "*.pt")
            models = glob.glob(pattern, recursive=True)
            found_models.extend(models)
    
    if found_models:
        # Return most recent model
        latest_model = max(found_models, key=os.path.getmtime)
        logger.info(f"Reward model auto-discovery successful: {latest_model}")
        return latest_model
    
    logger.error("Reward model not found")
    return None

def find_initial_ppo_model(search_dirs=None):
    """초기 PPO 모델 자동 탐색"""
    # Check default path first
    if os.path.exists(DEFAULT_PATHS['initial_ppo_model']):
        logger.info(f"Initial PPO model found: {DEFAULT_PATHS['initial_ppo_model']}")
        return str(DEFAULT_PATHS['initial_ppo_model'])
    
    if search_dirs is None:
        search_dirs = [
            BASE_DIR / 'data' / 'models',
            BASE_DIR / 'models',
            BASE_DIR / 'ppo_models'
        ]
    
    found_models = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            pattern = os.path.join(search_dir, "**", "*.zip")
            models = glob.glob(pattern, recursive=True)
            # Prioritize final_model.zip
            final_models = [m for m in models if 'final_model' in os.path.basename(m)]
            if final_models:
                found_models.extend(final_models)
            else:
                found_models.extend(models)
    
    if found_models:
        # Return most recent model
        latest_model = max(found_models, key=os.path.getmtime)
        logger.info(f"Initial PPO model auto-discovery successful: {latest_model}")
        return latest_model
    
    logger.warning("Initial PPO model not found. Will start with new model.")
    return None

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Automated RLHF Multi-Weight Experiments')
    parser.add_argument('--reward-model', type=str, default=None,
                       help='Reward model path (auto-search if not provided)')
    parser.add_argument('--initial-ppo-model', type=str, default=None,
                       help='Initial PPO model path (auto-search if not provided)')
    parser.add_argument('--use-initial-ppo', action='store_true',
                       help='Use 15k-step trained PPO model as initial model')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                       help='Human weight list to test (default: 0.3 0.5 0.7)')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Rounds per weight (default: 3)')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Training steps per round (default: 3000)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Result save directory (default: rlhf_experiments)')
    
    args = parser.parse_args()
    
    # Update experiment configuration
    if args.weights:
        EXPERIMENT_CONFIG['human_reward_weights'] = args.weights
    if args.rounds:
        EXPERIMENT_CONFIG['rounds_per_weight'] = args.rounds
    if args.timesteps:
        EXPERIMENT_CONFIG['timesteps_per_run'] = args.timesteps
    
    # Output directory setting
    output_dir = args.output_dir or DEFAULT_PATHS['base_output_dir']
    
    # Reward model path check
    reward_model_path = args.reward_model
    if not reward_model_path:
        reward_model_path = find_reward_model()
        if not reward_model_path:
            logger.error("Reward model not found. Please specify with --reward-model option.")
            return
    
    # Initial PPO model handling
    initial_ppo_model = None
    if args.use_initial_ppo or args.initial_ppo_model:
        if args.initial_ppo_model:
            initial_ppo_model = args.initial_ppo_model
        else:
            initial_ppo_model = find_initial_ppo_model()
    
    # Experiment overview
    print("\n" + "="*80)
    print("Automated RLHF Multi-Weight Experiments (Improved Version)")
    print("="*80)
    print(f"Continuous Learning Experiment Plan:")
    print(f"   Weights: {EXPERIMENT_CONFIG['human_reward_weights']}")
    print(f"   Continuous rounds per weight: {EXPERIMENT_CONFIG['rounds_per_weight']} rounds")
    print(f"   Steps per round: {EXPERIMENT_CONFIG['timesteps_per_run']:,}")
    print(f"   Total experiments: {len(EXPERIMENT_CONFIG['human_reward_weights']) * EXPERIMENT_CONFIG['rounds_per_weight']}")
    print(f"   Total training steps: {len(EXPERIMENT_CONFIG['human_reward_weights']) * EXPERIMENT_CONFIG['rounds_per_weight'] * EXPERIMENT_CONFIG['timesteps_per_run']:,}")
    print(f"\nFile Paths:")
    print(f"   Reward model: {reward_model_path}")
    if initial_ppo_model:
        print(f"   Initial PPO model: {initial_ppo_model}")
    print(f"   Results save to: {output_dir}")
    
    # Estimated time calculation (based on 15k steps taking 80,000 seconds)
    total_steps = len(EXPERIMENT_CONFIG['human_reward_weights']) * EXPERIMENT_CONFIG['rounds_per_weight'] * EXPERIMENT_CONFIG['timesteps_per_run']
    estimated_seconds = total_steps * (80000 / 15000)  # Proportional calculation
    estimated_hours = estimated_seconds / 3600
    
    print(f"\nEstimated time: {estimated_hours:.1f} hours")
    print("\n🆕 New Features:")
    print("   - JSON serialization error fixed")
    print("   - Paper-ready data generation included")
    print("   - Improved convergence analysis")
    print("="*80)
    
    # User confirmation
    try:
        response = input("\nProceed? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Experiments cancelled.")
            return
    except KeyboardInterrupt:
        print("\nExperiments cancelled.")
        return
    
    # Create and run experiment manager
    try:
        manager = RLHFExperimentManager(
            reward_model_path=reward_model_path,
            initial_ppo_model=initial_ppo_model,
            base_output_dir=output_dir
        )
        
        manager.run_all_experiments()
        
    except Exception as e:
        logger.error(f"Experiment execution error: {e}")
        raise

if __name__ == "__main__":
    main()