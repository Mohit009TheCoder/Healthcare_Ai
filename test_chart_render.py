#!/usr/bin/env python3
"""
Test script to check if chart data is being properly formatted
"""

from app import get_dashboard_charts
import json

charts = get_dashboard_charts()

# Test glucose_dist chart specifically
glucose_chart = charts.get('glucose_dist', {})

print("=" * 60)
print("Testing Glucose Distribution Chart")
print("=" * 60)

if glucose_chart:
    print(f"\n✅ Chart exists")
    print(f"Keys: {glucose_chart.keys()}")
    
    if 'data' in glucose_chart:
        print(f"\n📊 Data traces: {len(glucose_chart['data'])}")
        for i, trace in enumerate(glucose_chart['data']):
            print(f"\nTrace {i}:")
            print(f"  Type: {trace.get('type')}")
            print(f"  Name: {trace.get('name')}")
            if 'x' in trace:
                x_data = trace['x']
                print(f"  X data type: {type(x_data)}")
                print(f"  X data length: {len(x_data) if hasattr(x_data, '__len__') else 'N/A'}")
                print(f"  X data sample: {x_data[:5] if hasattr(x_data, '__getitem__') else x_data}")
            if 'y' in trace:
                y_data = trace['y']
                print(f"  Y data type: {type(y_data)}")
                print(f"  Y data length: {len(y_data) if hasattr(y_data, '__len__') else 'N/A'}")
    
    if 'layout' in glucose_chart:
        print(f"\n📐 Layout exists")
        layout = glucose_chart['layout']
        print(f"  Title: {layout.get('title')}")
        print(f"  Barmode: {layout.get('barmode')}")
    
    # Try to serialize to JSON
    print("\n🔍 JSON Serialization Test:")
    try:
        json_str = json.dumps(glucose_chart)
        print(f"  ✅ Success! JSON length: {len(json_str)} characters")
        # Save to file for inspection
        with open('/Users/mohitjain/Desktop/flask-healthcare/test_glucose_chart.json', 'w') as f:
            json.dump(glucose_chart, f, indent=2)
        print(f"  💾 Saved to test_glucose_chart.json")
    except Exception as e:
        print(f"  ❌ Error: {e}")
else:
    print("❌ Chart not found!")

print("\n" + "=" * 60)
