"""
Advanced Nutrition Analytics and Data Visualization Module
This module provides comprehensive nutritional analysis, trend tracking, and data visualization.

Features:
- Pandas for data analysis and manipulation
- Statistical analysis of eating patterns
- Trend detection and forecasting
- Data visualization with Matplotlib/Plotly
- Export capabilities for reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func, extract
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NutritionTrend:
    """Represents a nutritional trend over time."""
    metric: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    change_percentage: float
    confidence: float
    period: str
    data_points: List[Tuple[datetime, float]]

@dataclass
class HealthInsight:
    """Represents a health insight derived from data analysis."""
    category: str
    insight: str
    severity: str  # 'low', 'medium', 'high'
    recommendation: str
    data_support: Dict[str, Any]

@dataclass
class NutritionReport:
    """Comprehensive nutrition analysis report."""
    user_id: int
    period_start: datetime
    period_end: datetime
    total_calories: float
    average_daily_calories: float
    macro_distribution: Dict[str, float]
    trends: List[NutritionTrend]
    insights: List[HealthInsight]
    recommendations: List[str]
    charts_data: Dict[str, Any]

class NutritionAnalyzer:
    """
    Advanced nutrition analytics engine with comprehensive data analysis capabilities.
    
    This class provides:
    - Statistical analysis of nutritional data
    - Trend detection and forecasting
    - Health insights and recommendations
    - Data visualization and reporting
    """
    
    def __init__(self, db_session: Session):
        """Initialize the analyzer with database session."""
        self.db = db_session
        self.nutrition_goals = {
            'calories': 2000,
            'protein': 150,  # grams
            'carbs': 250,    # grams
            'fat': 65        # grams
        }
        
    def generate_comprehensive_report(self, user_id: int, days: int = 30) -> NutritionReport:
        """
        Generate a comprehensive nutrition analysis report.
        
        Args:
            user_id: User ID to analyze
            days: Number of days to analyze
            
        Returns:
            Complete nutrition report with insights and visualizations
        """
        logger.info(f"Generating nutrition report for user {user_id} over {days} days")
        
        # Get data
        meals_data = self._get_meals_data(user_id, days)
        if meals_data.empty:
            return self._empty_report(user_id, days)
        
        # Calculate metrics
        total_calories = meals_data['calories'].sum()
        avg_daily_calories = meals_data.groupby(meals_data['date'].dt.date)['calories'].sum().mean()
        macro_distribution = self._calculate_macro_distribution(meals_data)
        
        # Analyze trends
        trends = self._analyze_trends(meals_data)
        
        # Generate insights
        insights = self._generate_insights(meals_data, macro_distribution)
        
        # Create recommendations
        recommendations = self._generate_recommendations(insights, macro_distribution)
        
        # Prepare chart data
        charts_data = self._prepare_charts_data(meals_data)
        
        return NutritionReport(
            user_id=user_id,
            period_start=datetime.now() - timedelta(days=days),
            period_end=datetime.now(),
            total_calories=total_calories,
            average_daily_calories=avg_daily_calories,
            macro_distribution=macro_distribution,
            trends=trends,
            insights=insights,
            recommendations=recommendations,
            charts_data=charts_data
        )
    
    def _get_meals_data(self, user_id: int, days: int) -> pd.DataFrame:
        """Retrieve and process meals data from database."""
        from models import Meal
        
        # Query meals from database
        meals = self.db.query(Meal).filter(
            Meal.user_id == user_id,
            Meal.created_at >= datetime.now() - timedelta(days=days)
        ).all()
        
        if not meals:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for meal in meals:
            data.append({
                'id': meal.id,
                'name': meal.name,
                'calories': meal.calories,
                'protein': meal.protein,
                'carbs': meal.carbs,
                'fat': meal.fat,
                'date': meal.created_at,
                'description': meal.description
            })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        df['meal_type'] = df['hour'].apply(self._classify_meal_type)
        
        return df
    
    def _classify_meal_type(self, hour: int) -> str:
        """Classify meal type based on hour of day."""
        if 5 <= hour < 11:
            return 'Breakfast'
        elif 11 <= hour < 15:
            return 'Lunch'
        elif 15 <= hour < 18:
            return 'Snack'
        else:
            return 'Dinner'
    
    def _calculate_macro_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate macro nutrient distribution."""
        total_calories = df['calories'].sum()
        if total_calories == 0:
            return {'protein': 0, 'carbs': 0, 'fat': 0}
        
        # Convert grams to calories (protein: 4 cal/g, carbs: 4 cal/g, fat: 9 cal/g)
        protein_calories = df['protein'].sum() * 4
        carbs_calories = df['carbs'].sum() * 4
        fat_calories = df['fat'].sum() * 9
        
        return {
            'protein': round((protein_calories / total_calories) * 100, 1),
            'carbs': round((carbs_calories / total_calories) * 100, 1),
            'fat': round((fat_calories / total_calories) * 100, 1)
        }
    
    def _analyze_trends(self, df: pd.DataFrame) -> List[NutritionTrend]:
        """Analyze nutritional trends over time."""
        trends = []
        
        # Daily aggregation
        daily_data = df.groupby(df['date'].dt.date).agg({
            'calories': 'sum',
            'protein': 'sum',
            'carbs': 'sum',
            'fat': 'sum'
        }).reset_index()
        
        if len(daily_data) < 7:  # Need at least a week of data
            return trends
        
        # Analyze each metric
        for metric in ['calories', 'protein', 'carbs', 'fat']:
            trend = self._calculate_trend(daily_data, metric)
            if trend:
                trends.append(trend)
        
        return trends
    
    def _calculate_trend(self, daily_data: pd.DataFrame, metric: str) -> Optional[NutritionTrend]:
        """Calculate trend for a specific metric."""
        values = daily_data[metric].values
        dates = daily_data['date'].values
        
        if len(values) < 3:
            return None
        
        # Simple linear regression for trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Calculate change percentage
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        if len(first_half) == 0 or len(second_half) == 0:
            return None
        
        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)
        
        if first_avg == 0:
            change_percentage = 0
        else:
            change_percentage = ((second_avg - first_avg) / first_avg) * 100
        
        # Determine trend direction
        if abs(change_percentage) < 5:
            direction = 'stable'
        elif change_percentage > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Calculate confidence based on data consistency
        confidence = min(1.0, len(values) / 30)  # More data = higher confidence
        
        return NutritionTrend(
            metric=metric,
            trend_direction=direction,
            change_percentage=round(change_percentage, 1),
            confidence=round(confidence, 2),
            period=f"{len(values)} days",
            data_points=list(zip(dates, values))
        )
    
    def _generate_insights(self, df: pd.DataFrame, macro_dist: Dict[str, float]) -> List[HealthInsight]:
        """Generate health insights from the data."""
        insights = []
        
        # Calorie intake analysis
        daily_calories = df.groupby(df['date'].dt.date)['calories'].sum()
        avg_calories = daily_calories.mean()
        
        if avg_calories < self.nutrition_goals['calories'] * 0.8:
            insights.append(HealthInsight(
                category="Calorie Intake",
                insight=f"Average daily calorie intake ({avg_calories:.0f}) is below recommended levels",
                severity="medium",
                recommendation="Consider increasing portion sizes or adding healthy snacks",
                data_support={"avg_calories": avg_calories, "goal": self.nutrition_goals['calories']}
            ))
        elif avg_calories > self.nutrition_goals['calories'] * 1.2:
            insights.append(HealthInsight(
                category="Calorie Intake",
                insight=f"Average daily calorie intake ({avg_calories:.0f}) exceeds recommended levels",
                severity="high",
                recommendation="Consider reducing portion sizes or choosing lower-calorie options",
                data_support={"avg_calories": avg_calories, "goal": self.nutrition_goals['calories']}
            ))
        
        # Macro distribution analysis
        if macro_dist['protein'] < 15:
            insights.append(HealthInsight(
                category="Protein Intake",
                insight=f"Protein intake ({macro_dist['protein']}%) is below recommended 15-20%",
                severity="medium",
                recommendation="Add more lean proteins like chicken, fish, or legumes",
                data_support={"protein_percentage": macro_dist['protein']}
            ))
        
        if macro_dist['fat'] > 35:
            insights.append(HealthInsight(
                category="Fat Intake",
                insight=f"Fat intake ({macro_dist['fat']}%) exceeds recommended 20-35%",
                severity="medium",
                recommendation="Choose leaner protein sources and reduce added fats",
                data_support={"fat_percentage": macro_dist['fat']}
            ))
        
        # Meal timing analysis
        meal_times = df.groupby('meal_type')['calories'].sum()
        if 'Breakfast' in meal_times and meal_times['Breakfast'] < meal_times.sum() * 0.2:
            insights.append(HealthInsight(
                category="Meal Timing",
                insight="Breakfast calories are low compared to other meals",
                severity="low",
                recommendation="Consider a more substantial breakfast to fuel your day",
                data_support={"breakfast_percentage": (meal_times['Breakfast'] / meal_times.sum()) * 100}
            ))
        
        return insights
    
    def _generate_recommendations(self, insights: List[HealthInsight], macro_dist: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on insights."""
        recommendations = []
        
        # General recommendations
        if macro_dist['protein'] < 15:
            recommendations.append("Increase protein intake by adding lean meats, fish, eggs, or plant-based proteins")
        
        if macro_dist['carbs'] > 60:
            recommendations.append("Consider reducing refined carbohydrates and increasing whole grains")
        
        if macro_dist['fat'] > 35:
            recommendations.append("Choose leaner protein sources and reduce added oils and fats")
        
        # Specific recommendations from insights
        for insight in insights:
            if insight.severity in ['high', 'medium']:
                recommendations.append(insight.recommendation)
        
        # Add general health recommendations
        recommendations.extend([
            "Aim for consistent meal timing to support metabolism",
            "Include a variety of colorful fruits and vegetables daily",
            "Stay hydrated with water throughout the day",
            "Consider meal prep to maintain consistent nutrition"
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _prepare_charts_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for various charts and visualizations."""
        charts_data = {}
        
        # Daily calories chart
        daily_calories = df.groupby(df['date'].dt.date)['calories'].sum().reset_index()
        charts_data['daily_calories'] = {
            'dates': daily_calories['date'].astype(str).tolist(),
            'calories': daily_calories['calories'].tolist()
        }
        
        # Macro distribution pie chart
        macro_dist = self._calculate_macro_distribution(df)
        charts_data['macro_distribution'] = {
            'labels': ['Protein', 'Carbs', 'Fat'],
            'values': [macro_dist['protein'], macro_dist['carbs'], macro_dist['fat']]
        }
        
        # Meal timing analysis
        meal_timing = df.groupby('meal_type')['calories'].sum().reset_index()
        charts_data['meal_timing'] = {
            'meal_types': meal_timing['meal_type'].tolist(),
            'calories': meal_timing['calories'].tolist()
        }
        
        # Weekly pattern
        df['weekday'] = df['date'].dt.day_name()
        weekly_pattern = df.groupby('weekday')['calories'].sum().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]).fillna(0)
        charts_data['weekly_pattern'] = {
            'days': weekly_pattern.index.tolist(),
            'calories': weekly_pattern.values.tolist()
        }
        
        return charts_data
    
    def _empty_report(self, user_id: int, days: int) -> NutritionReport:
        """Return empty report when no data is available."""
        return NutritionReport(
            user_id=user_id,
            period_start=datetime.now() - timedelta(days=days),
            period_end=datetime.now(),
            total_calories=0,
            average_daily_calories=0,
            macro_distribution={'protein': 0, 'carbs': 0, 'fat': 0},
            trends=[],
            insights=[],
            recommendations=["Start logging your meals to get personalized insights!"],
            charts_data={}
        )
    
    def create_visualization_dashboard(self, report: NutritionReport) -> Dict[str, str]:
        """
        Create interactive visualization dashboard using Plotly.
        
        Returns:
            Dictionary with chart HTML strings
        """
        charts = {}
        
        # Daily calories trend
        if 'daily_calories' in report.charts_data:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=report.charts_data['daily_calories']['dates'],
                y=report.charts_data['daily_calories']['calories'],
                mode='lines+markers',
                name='Daily Calories',
                line=dict(color='#ff6b35', width=3)
            ))
            fig.add_hline(
                y=self.nutrition_goals['calories'],
                line_dash="dash",
                line_color="red",
                annotation_text="Daily Goal"
            )
            fig.update_layout(
                title="Daily Calorie Intake Trend",
                xaxis_title="Date",
                yaxis_title="Calories",
                template="plotly_dark"
            )
            charts['daily_trend'] = fig.to_html(include_plotlyjs=False, div_id="daily-trend")
        
        # Macro distribution pie chart
        if 'macro_distribution' in report.charts_data:
            fig = go.Figure(data=[go.Pie(
                labels=report.charts_data['macro_distribution']['labels'],
                values=report.charts_data['macro_distribution']['values'],
                hole=0.3,
                marker_colors=['#ff6b35', '#3498db', '#2ecc71']
            )])
            fig.update_layout(
                title="Macro Nutrient Distribution",
                template="plotly_dark"
            )
            charts['macro_pie'] = fig.to_html(include_plotlyjs=False, div_id="macro-pie")
        
        # Meal timing bar chart
        if 'meal_timing' in report.charts_data:
            fig = go.Figure(data=[go.Bar(
                x=report.charts_data['meal_timing']['meal_types'],
                y=report.charts_data['meal_timing']['calories'],
                marker_color='#3498db'
            )])
            fig.update_layout(
                title="Calories by Meal Type",
                xaxis_title="Meal Type",
                yaxis_title="Calories",
                template="plotly_dark"
            )
            charts['meal_timing'] = fig.to_html(include_plotlyjs=False, div_id="meal-timing")
        
        # Weekly pattern
        if 'weekly_pattern' in report.charts_data:
            fig = go.Figure(data=[go.Bar(
                x=report.charts_data['weekly_pattern']['days'],
                y=report.charts_data['weekly_pattern']['calories'],
                marker_color='#2ecc71'
            )])
            fig.update_layout(
                title="Weekly Calorie Pattern",
                xaxis_title="Day of Week",
                yaxis_title="Calories",
                template="plotly_dark"
            )
            charts['weekly_pattern'] = fig.to_html(include_plotlyjs=False, div_id="weekly-pattern")
        
        return charts
    
    def export_report_to_json(self, report: NutritionReport, filepath: str) -> None:
        """Export nutrition report to JSON file."""
        report_dict = {
            'user_id': report.user_id,
            'period_start': report.period_start.isoformat(),
            'period_end': report.period_end.isoformat(),
            'total_calories': report.total_calories,
            'average_daily_calories': report.average_daily_calories,
            'macro_distribution': report.macro_distribution,
            'trends': [
                {
                    'metric': t.metric,
                    'trend_direction': t.trend_direction,
                    'change_percentage': t.change_percentage,
                    'confidence': t.confidence,
                    'period': t.period
                } for t in report.trends
            ],
            'insights': [
                {
                    'category': i.category,
                    'insight': i.insight,
                    'severity': i.severity,
                    'recommendation': i.recommendation,
                    'data_support': i.data_support
                } for i in report.insights
            ],
            'recommendations': report.recommendations,
            'charts_data': report.charts_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Report exported to {filepath}")

# Example usage
if __name__ == "__main__":
    # This would be used with an actual database session
    # from database import get_db
    # db = next(get_db())
    # analyzer = NutritionAnalyzer(db)
    # report = analyzer.generate_comprehensive_report(user_id=1, days=30)
    # charts = analyzer.create_visualization_dashboard(report)
    # analyzer.export_report_to_json(report, "nutrition_report.json")
    pass

