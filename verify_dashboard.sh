#!/bin/bash

echo "=========================================="
echo "Dashboard Verification Script"
echo "=========================================="
echo ""

echo "1. Testing chart generation..."
python3 test_dashboard.py 2>&1 | grep -E "(Generated|✅|❌)" | head -20

echo ""
echo "2. Checking chart data structure..."
python3 test_chart_render.py 2>&1 | grep -E "(✅|❌|Success|Error)" | head -10

echo ""
echo "3. Verifying files exist..."
files=(
    "app.py"
    "templates/dashboard.html"
    "static/js/main.js"
    "DASHBOARD_IMPROVEMENTS.md"
    "FIXES_APPLIED.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
    fi
done

echo ""
echo "4. Checking for required dependencies..."
python3 -c "import plotly; print('✅ Plotly version:', plotly.__version__)" 2>/dev/null || echo "❌ Plotly not installed"
python3 -c "import flask; print('✅ Flask version:', flask.__version__)" 2>/dev/null || echo "❌ Flask not installed"
python3 -c "import pandas; print('✅ Pandas version:', pandas.__version__)" 2>/dev/null || echo "❌ Pandas not installed"

echo ""
echo "=========================================="
echo "Verification Complete!"
echo "=========================================="
echo ""
echo "To start the application:"
echo "  python3 app.py"
echo ""
echo "Then visit:"
echo "  http://localhost:5001/dashboard"
echo ""
