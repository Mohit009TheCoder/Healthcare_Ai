#!/usr/bin/env python3
"""
Test script to verify dashboard charts are working correctly
"""

from app import get_dashboard_charts
import json

def test_charts():
    print("=" * 60)
    print("Testing Dashboard Charts")
    print("=" * 60)
    
    charts = get_dashboard_charts()
    
    print(f"\n✅ Generated {len(charts)} charts\n")
    
    for name, chart in charts.items():
        if chart and 'data' in chart and 'layout' in chart:
            num_traces = len(chart['data'])
            chart_type = chart['data'][0].get('type', 'unknown') if chart['data'] else 'no data'
            title = chart['layout'].get('title', {}).get('text', 'No title') if isinstance(chart['layout'].get('title'), dict) else chart['layout'].get('title', 'No title')
            
            print(f"✅ {name:20s} | Traces: {num_traces} | Type: {chart_type:12s} | Title: {title}")
            
            # Verify JSON serializable
            try:
                json.dumps(chart)
                print(f"   └─ JSON: ✅ Serializable")
            except Exception as e:
                print(f"   └─ JSON: ❌ Error: {e}")
        else:
            print(f"❌ {name:20s} | EMPTY OR INVALID")
        print()
    
    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == '__main__':
    test_charts()
