"""
Response formatter for air quality chatbot
Formats RAG outputs, generates evidence-based health recommendations, and computes confidence scores.
"""

from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Formats responses from the RAG pipeline with robust type safety"""
    
    def format_response(self, response: str, context: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final response"""
        if user_context is None:
            user_context = {}
            
        # Extract data sources from context
        data_sources = self._extract_data_sources(context)
        
        # Generate recommendations based on user context and environmental context
        recommendations = self._generate_recommendations(user_context, context)
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(context, data_sources)
        
        return {
            'response': response,
            'confidence': confidence,
            'data_sources': data_sources,
            'timestamp': datetime.utcnow().isoformat(),
            'recommendations': recommendations
        }
    
    def _extract_data_sources(self, context: str) -> List[str]:
        """Extract data sources from context"""
        sources = []
        ctx_lower = context.lower() if context else ""
        
        if 'openaq' in ctx_lower:
            sources.append('OpenAQ API')
        if 'weather' in ctx_lower or 'inversion' in ctx_lower or 'meteorological' in ctx_lower:
            sources.append('Weather API')
        if 'who' in ctx_lower:
            sources.append('WHO Air Quality Guidelines 2021')
        if 'epa' in ctx_lower or 'aqi' in ctx_lower:
            sources.append('EPA Standards')
        if 'nasa' in ctx_lower:
            sources.append('NASA Earth Observations')
            
        return sources if sources else ['WHO/EPA Guidelines', 'OpenAQ']
    
    def _has_condition(self, user_context: Dict[str, Any], condition_name: str) -> bool:
        """Safely check if a condition exists across dict, string, or list contexts"""
        if not user_context:
            return False
            
        # Check 'profile' or 'group'
        for key in ['profile', 'group', 'vulnerability', 'condition']:
            val = str(user_context.get(key, '')).lower()
            if condition_name in val:
                return True
                
        # Check health_conditions
        health = user_context.get('health_conditions')
        if isinstance(health, dict):
            return bool(health.get(condition_name))
        elif isinstance(health, (list, tuple, set)):
            return any(condition_name in str(item).lower() for item in health)
        elif isinstance(health, str):
            return condition_name in health.lower()
            
        return False
    
    def _generate_recommendations(self, user_context: Dict[str, Any], context: str) -> List[str]:
        """Generate personalized health recommendations safely"""
        recommendations = []
        ctx_lower = context.lower() if context else ""
        
        is_poor_air = any(w in ctx_lower for w in ['unhealthy', 'poor', 'hazardous', 'moderate', 'severe', 'inversion'])
        
        # Check Asthma / Respiratory
        if self._has_condition(user_context, 'asthma') or self._has_condition(user_context, 'respiratory'):
            recommendations.append("Asthma & Respiratory: Keep fast-acting rescue inhalers immediately accessible.")
            recommendations.append("Stay in rooms equipped with verified HEPA air filtration when AQI exceeds 100.")
        
        # Check Children
        age_group = str(user_context.get('age_group', '')).lower()
        if age_group == 'children' or self._has_condition(user_context, 'child') or self._has_condition(user_context, 'school'):
            recommendations.append("Children: Substitute prolonged outdoor cardio play with indoor activities when AQI > 100.")
            recommendations.append("Ensure indoor play areas and classrooms maintain clean air filtration.")
        
        # Check Elderly
        if age_group == 'elderly' or self._has_condition(user_context, 'elderly') or self._has_condition(user_context, 'senior'):
            recommendations.append("Elderly individuals: Minimize strenuous outdoor physical exertion during peak pollution.")
            recommendations.append("Monitor for early symptoms of respiratory strain or chest tightness.")
            
        # Check Athletes / Runners
        if self._has_condition(user_context, 'runner') or self._has_condition(user_context, 'athlete'):
            if is_poor_air:
                recommendations.append("Athletes: Shift high-intensity aerobic workouts indoors to avoid deep particulate inhalation.")
            else:
                recommendations.append("Outdoor exercise is safe; schedule long endurance runs during morning low-traffic hours.")
        
        # Check Pregnancy
        if self._has_condition(user_context, 'pregnancy') or self._has_condition(user_context, 'pregnant'):
            recommendations.append("Expectant mothers: Avoid congested traffic corridors and run indoor air purifiers.")
        
        # General Environmental Protections
        if 'hazardous' in ctx_lower or 'very unhealthy' in ctx_lower:
            recommendations.append("Emergency Alert: Keep all doors and windows sealed; use continuous air purifiers.")
            recommendations.append("Wear a fitted NIOSH-approved N95/FFP2 respirator if outdoor transit is essential.")
        elif 'unhealthy' in ctx_lower:
            recommendations.append("General Public: Reduce prolonged outdoor exertion; keep residential windows closed.")
            recommendations.append("Operate HVAC ventilation on recirculation mode.")
        elif not recommendations:
            recommendations.append("Air quality is favorable: Safe for open ventilation and normal outdoor recreational activities.")
            recommendations.append("Regularly check daily air forecasts to plan outdoor exertion.")
        
        # Return unique recommendations, max 5
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)
                if len(unique_recs) >= 5:
                    break
                    
        return unique_recs
    
    def _calculate_confidence(self, context: str, data_sources: List[str]) -> float:
        """Calculate confidence score based on data coverage and quality"""
        confidence = 0.70  # Solid baseline confidence
        
        if len(data_sources) >= 2:
            confidence += 0.15
        if 'OpenAQ' in str(data_sources):
            confidence += 0.05
        if 'WHO' in str(data_sources) or 'EPA' in str(data_sources):
            confidence += 0.05
            
        return round(min(confidence, 0.98), 2)
