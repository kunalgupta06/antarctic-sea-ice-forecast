"""
Sea Level Rise Impact Calculator
=================================
Connects Antarctic ice loss to global sea level rise and coastal city impacts

This module:
1. Converts ice loss predictions to sea level rise
2. Maps impacts to specific coastal cities
3. Calculates population and economic exposure
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class CoastalCity:
    """Data for a coastal city at risk"""
    name: str
    country: str
    population: int
    current_elevation_m: float
    assets_at_risk_billion_usd: float
    population_at_risk: int
    critical_slr_threshold_m: float  # Sea level rise that causes major flooding


# Major cities at risk (based on research)
COASTAL_CITIES = [
    CoastalCity("Miami", "USA", 6_200_000, 1.8, 3500, 2_500_000, 0.6),
    CoastalCity("New York", "USA", 8_300_000, 3.0, 2200, 1_800_000, 1.0),
    CoastalCity("New Orleans", "USA", 390_000, -0.5, 500, 350_000, 0.3),
    CoastalCity("Shanghai", "China", 24_000_000, 4.0, 1800, 5_000_000, 1.5),
    CoastalCity("Mumbai", "India", 21_000_000, 8.0, 1200, 11_000_000, 0.5),
    CoastalCity("Kolkata", "India", 14_800_000, 9.0, 800, 14_000_000, 0.4),
    CoastalCity("Dhaka", "Bangladesh", 22_000_000, 4.0, 600, 11_000_000, 0.5),
    CoastalCity("Bangkok", "Thailand", 10_500_000, 1.5, 900, 5_400_000, 0.4),
    CoastalCity("Tokyo", "Japan", 13_900_000, 5.0, 2500, 3_200_000, 1.2),
    CoastalCity("Jakarta", "Indonesia", 10_500_000, 2.0, 700, 4_000_000, 0.5),
    CoastalCity("Lagos", "Nigeria", 15_400_000, 3.0, 400, 6_000_000, 0.6),
    CoastalCity("Boston", "USA", 675_000, 2.5, 450, 180_000, 0.8),
    CoastalCity("London", "UK", 8_800_000, 5.0, 1500, 1_200_000, 1.5),
    CoastalCity("Amsterdam", "Netherlands", 870_000, -2.0, 600, 500_000, 0.3),
    CoastalCity("Venice", "Italy", 260_000, 1.0, 200, 200_000, 0.2),
]


class SeaLevelImpactCalculator:
    """
    Calculates sea level rise impacts based on Antarctic ice loss predictions
    """
    
    # Constants from scientific research
    ANTARCTIC_TOTAL_ICE_SLR_METERS = 58.0  # If all Antarctic ice melted
    WEST_ANTARCTIC_SLR_METERS = 5.3  # West Antarctic Ice Sheet alone
    ANNUAL_ICE_LOSS_GIGATONS = 150  # Current annual loss
    
    # Conversion: 362.5 Gt of ice = 1mm sea level rise
    GT_TO_MM_SLR = 1 / 362.5
    
    def __init__(self):
        self.cities = COASTAL_CITIES
    
    def ice_extent_to_sea_level_rise(
        self,
        ice_extent_change_percent: float,
        years: int = 50
    ) -> Dict[str, float]:
        """
        Convert ice extent change to sea level rise
        
        Args:
            ice_extent_change_percent: % change in ice extent (negative = loss)
            years: Number of years for projection
        
        Returns:
            Dict with sea level rise projections
        """
        # Approximate relationship between extent and volume
        # (extent is 2D, but correlates with volume changes)
        volume_change_factor = ice_extent_change_percent / 100 * 1.5  # Amplification factor
        
        # Calculate SLR contribution (simplified)
        # West Antarctic is most vulnerable, contributes ~5.3m if fully melted
        west_antarctic_contribution = abs(volume_change_factor) * self.WEST_ANTARCTIC_SLR_METERS
        
        # Add baseline annual melt contribution
        baseline_contribution_mm = self.ANNUAL_ICE_LOSS_GIGATONS * self.GT_TO_MM_SLR * years
        baseline_contribution_m = baseline_contribution_mm / 1000
        
        total_slr = baseline_contribution_m + west_antarctic_contribution * 0.1  # Conservative estimate
        
        return {
            'total_sea_level_rise_m': total_slr,
            'total_sea_level_rise_cm': total_slr * 100,
            'total_sea_level_rise_inches': total_slr * 39.37,
            'baseline_contribution_m': baseline_contribution_m,
            'ice_loss_contribution_m': west_antarctic_contribution * 0.1,
            'years': years,
            'scenario': self._categorize_scenario(total_slr)
        }
    
    def _categorize_scenario(self, slr_meters: float) -> str:
        """Categorize the sea level rise scenario"""
        if slr_meters < 0.3:
            return "Low"
        elif slr_meters < 0.6:
            return "Moderate"
        elif slr_meters < 1.0:
            return "High"
        else:
            return "Extreme"
    
    def calculate_city_impacts(
        self,
        sea_level_rise_m: float
    ) -> List[Dict]:
        """
        Calculate impacts for each coastal city
        
        Args:
            sea_level_rise_m: Projected sea level rise in meters
        
        Returns:
            List of city impact assessments
        """
        impacts = []
        
        for city in self.cities:
            # Calculate flood risk increase
            # Every 10cm triples flood frequency
            flood_frequency_multiplier = 3 ** (sea_level_rise_m / 0.1)
            
            # Coastline retreat (1cm SLR = ~1m retreat)
            coastline_retreat_m = sea_level_rise_m * 100
            
            # Risk level based on threshold
            if sea_level_rise_m >= city.critical_slr_threshold_m:
                risk_level = "CRITICAL"
                impact_severity = 1.0
            elif sea_level_rise_m >= city.critical_slr_threshold_m * 0.7:
                risk_level = "HIGH"
                impact_severity = 0.7
            elif sea_level_rise_m >= city.critical_slr_threshold_m * 0.4:
                risk_level = "MODERATE"
                impact_severity = 0.4
            else:
                risk_level = "LOW"
                impact_severity = 0.2
            
            # Economic impact (simplified)
            economic_impact = city.assets_at_risk_billion_usd * impact_severity
            
            # Population affected
            population_affected = int(city.population_at_risk * impact_severity)
            
            impacts.append({
                'city': city.name,
                'country': city.country,
                'population': city.population,
                'risk_level': risk_level,
                'sea_level_rise_m': sea_level_rise_m,
                'critical_threshold_m': city.critical_slr_threshold_m,
                'flood_frequency_increase': f"{flood_frequency_multiplier:.0f}x",
                'coastline_retreat_m': coastline_retreat_m,
                'economic_impact_billion_usd': economic_impact,
                'population_affected': population_affected,
                'percent_of_city_affected': (population_affected / city.population) * 100
            })
        
        # Sort by risk level
        risk_order = {'CRITICAL': 0, 'HIGH': 1, 'MODERATE': 2, 'LOW': 3}
        impacts.sort(key=lambda x: (risk_order[x['risk_level']], -x['population_affected']))
        
        return impacts
    
    def generate_impact_report(
        self,
        ice_extent_predictions: np.ndarray,
        years: int = 50
    ) -> Dict:
        """
        Generate comprehensive impact report from ice predictions
        
        Args:
            ice_extent_predictions: Array of ice extent values (0-1)
            years: Number of years predicted
        
        Returns:
            Comprehensive impact report
        """
        # Calculate ice extent change
        initial_extent = ice_extent_predictions[0]
        final_extent = ice_extent_predictions[-1]
        extent_change_percent = ((final_extent - initial_extent) / initial_extent) * 100
        
        # Get sea level rise
        slr = self.ice_extent_to_sea_level_rise(extent_change_percent, years)
        
        # Get city impacts
        city_impacts = self.calculate_city_impacts(slr['total_sea_level_rise_m'])
        
        # Summary statistics
        total_population_affected = sum(c['population_affected'] for c in city_impacts)
        total_economic_impact = sum(c['economic_impact_billion_usd'] for c in city_impacts)
        critical_cities = [c for c in city_impacts if c['risk_level'] == 'CRITICAL']
        high_risk_cities = [c for c in city_impacts if c['risk_level'] == 'HIGH']
        
        return {
            'summary': {
                'projection_years': years,
                'ice_extent_change_percent': extent_change_percent,
                'sea_level_rise_m': slr['total_sea_level_rise_m'],
                'sea_level_rise_cm': slr['total_sea_level_rise_cm'],
                'scenario': slr['scenario'],
                'total_population_affected': total_population_affected,
                'total_economic_impact_billion_usd': total_economic_impact,
                'critical_cities_count': len(critical_cities),
                'high_risk_cities_count': len(high_risk_cities),
            },
            'sea_level_rise_details': slr,
            'city_impacts': city_impacts,
            'critical_cities': [c['city'] for c in critical_cities],
            'key_findings': self._generate_key_findings(slr, city_impacts)
        }
    
    def _generate_key_findings(self, slr: Dict, city_impacts: List[Dict]) -> List[str]:
        """Generate key findings for the report"""
        findings = []
        
        # Sea level finding
        findings.append(
            f"Sea level is projected to rise {slr['total_sea_level_rise_cm']:.1f} cm "
            f"({slr['total_sea_level_rise_inches']:.1f} inches) over {slr['years']} years"
        )
        
        # Most affected cities
        top_3 = city_impacts[:3]
        findings.append(
            f"Most vulnerable cities: {', '.join(c['city'] for c in top_3)}"
        )
        
        # Population impact
        total_pop = sum(c['population_affected'] for c in city_impacts)
        findings.append(
            f"Approximately {total_pop / 1_000_000:.1f} million people could be affected"
        )
        
        # Economic impact
        total_econ = sum(c['economic_impact_billion_usd'] for c in city_impacts)
        findings.append(
            f"Potential economic impact: ${total_econ:.0f} billion in at-risk assets"
        )
        
        # Flood frequency
        avg_flood_increase = np.mean([
            float(c['flood_frequency_increase'].replace('x', '')) 
            for c in city_impacts
        ])
        findings.append(
            f"Coastal flooding frequency could increase {avg_flood_increase:.0f}x on average"
        )
        
        return findings


def print_impact_report(report: Dict):
    """Pretty print the impact report"""
    print("\n" + "=" * 70)
    print("🌊 SEA LEVEL RISE IMPACT ASSESSMENT")
    print("=" * 70)
    
    summary = report['summary']
    print(f"\n📊 SUMMARY")
    print(f"   Projection Period: {summary['projection_years']} years")
    print(f"   Ice Extent Change: {summary['ice_extent_change_percent']:.1f}%")
    print(f"   Sea Level Rise: {summary['sea_level_rise_cm']:.1f} cm ({summary['sea_level_rise_m']:.2f} m)")
    print(f"   Scenario: {summary['scenario']}")
    
    print(f"\n🏙️ GLOBAL IMPACT")
    print(f"   Population Affected: {summary['total_population_affected'] / 1_000_000:.1f} million")
    print(f"   Economic Risk: ${summary['total_economic_impact_billion_usd']:.0f} billion")
    print(f"   Critical Cities: {summary['critical_cities_count']}")
    print(f"   High-Risk Cities: {summary['high_risk_cities_count']}")
    
    print(f"\n🔴 MOST VULNERABLE CITIES")
    print("-" * 70)
    print(f"{'City':<20} {'Country':<12} {'Risk':<10} {'Pop. Affected':<15} {'Economic ($B)':<12}")
    print("-" * 70)
    
    for city in report['city_impacts'][:10]:
        print(f"{city['city']:<20} {city['country']:<12} {city['risk_level']:<10} "
              f"{city['population_affected']/1_000_000:.2f}M{'':<8} "
              f"${city['economic_impact_billion_usd']:.0f}B")
    
    print(f"\n📋 KEY FINDINGS")
    for i, finding in enumerate(report['key_findings'], 1):
        print(f"   {i}. {finding}")
    
    print("\n" + "=" * 70)


# Example usage
if __name__ == "__main__":
    calculator = SeaLevelImpactCalculator()
    
    # Simulate ice extent predictions (declining from 55% to 35% over 50 years)
    years = 50
    ice_predictions = np.linspace(0.55, 0.35, years * 12)  # Monthly data
    
    # Generate report
    report = calculator.generate_impact_report(ice_predictions, years)
    
    # Print report
    print_impact_report(report)