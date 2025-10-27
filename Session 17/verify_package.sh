#!/bin/bash
# verify_package.sh
# Verify Session 17 deliverables are complete

echo "════════════════════════════════════════════════════════════"
echo "  SPECTRA-Lab Session 17 - Package Verification"
echo "════════════════════════════════════════════════════════════"
echo ""

MISSING=0

check_file() {
    if [ -f "$1" ]; then
        echo "✅ $1"
    else
        echo "❌ MISSING: $1"
        MISSING=$((MISSING+1))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo "✅ $1/"
    else
        echo "❌ MISSING: $1/"
        MISSING=$((MISSING+1))
    fi
}

echo "📋 Checking Documentation..."
check_file "README.md"
check_file "SESSION_17.md"
check_file "QUICKSTART.md"
check_file "DELIVERY_SUMMARY.md"
check_file ".env.example"

echo ""
echo "🗄️ Checking Database Files..."
check_file "alembic.ini"
check_dir "alembic"
check_file "alembic/env.py"
check_file "seed_demo.py"

echo ""
echo "🔧 Checking Backend Implementation..."
check_dir "services/shared"
check_file "services/shared/db/base.py"
check_file "services/shared/db/models.py"
check_file "services/shared/db/deps.py"
check_file "services/shared/auth/jwt.py"

echo ""
echo "🧪 Checking Tests..."
check_file "test_session17.py"

echo ""
echo "🚀 Checking Deployment Files..."
check_file "deploy_session17.sh"
check_file "docker-compose.yml"

echo ""
echo "════════════════════════════════════════════════════════════"
if [ $MISSING -eq 0 ]; then
    echo "✅ Package verification PASSED"
    echo "   All deliverables present and accounted for!"
    echo ""
    echo "📦 Package Contents:"
    echo "   • Documentation:    5 files"
    echo "   • Database:         3 files + 1 directory"
    echo "   • Backend Code:     4 implementation files"
    echo "   • Tests:            1 integration test suite"
    echo "   • Deployment:       2 automation files"
    echo ""
    echo "🎯 Ready for deployment!"
    exit 0
else
    echo "❌ Package verification FAILED"
    echo "   Missing $MISSING files or directories"
    echo ""
    echo "Please ensure all deliverables are extracted properly."
    exit 1
fi
