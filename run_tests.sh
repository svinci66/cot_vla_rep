#!/bin/bash
# 运行所有测试脚本的便捷脚本

echo "=========================================="
echo "VILA-U Action Prediction - Test Runner"
echo "=========================================="
echo ""

# 设置颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试计数
TOTAL=0
PASSED=0
FAILED=0

# 运行单个测试
run_test() {
    local test_file=$1
    local test_name=$2

    echo -e "${YELLOW}Running: ${test_name}${NC}"
    echo "----------------------------------------"

    TOTAL=$((TOTAL + 1))

    if python "$test_file"; then
        echo -e "${GREEN}✓ ${test_name} PASSED${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ ${test_name} FAILED${NC}"
        FAILED=$((FAILED + 1))
    fi

    echo ""
}

# 运行所有测试
echo "Starting test suite..."
echo ""

# Step 1: Constants and Configuration
if [ -f "tests/test_step1_constants.py" ]; then
    run_test "tests/test_step1_constants.py" "Step 1: Constants and Configuration"
fi

# Step 2: Action Head
if [ -f "tests/test_step2_action_head.py" ]; then
    run_test "tests/test_step2_action_head.py" "Step 2: Action Prediction Head"
fi

# Step 3: Data Loader
if [ -f "tests/test_step3_data_loader.py" ]; then
    run_test "tests/test_step3_data_loader.py" "Step 3: LIBERO Data Loader"
fi

# Step 4: Training (if exists)
if [ -f "tests/test_step4_training.py" ]; then
    run_test "tests/test_step4_training.py" "Step 4: Training Logic"
fi

# Step 5: Inference (if exists)
if [ -f "tests/test_step5_inference.py" ]; then
    run_test "tests/test_step5_inference.py" "Step 5: Inference Interface"
fi

# Step 6: Trajectory (if exists)
if [ -f "tests/test_step6_trajectory.py" ]; then
    run_test "tests/test_step6_trajectory.py" "Step 6: Trajectory Generation"
fi

# Step 7: Save Format (if exists)
if [ -f "tests/test_step7_save_format.py" ]; then
    run_test "tests/test_step7_save_format.py" "Step 7: LIBERO Format Saving"
fi

# Step 8: End-to-End (if exists)
if [ -f "tests/test_step8_end_to_end.py" ]; then
    run_test "tests/test_step8_end_to_end.py" "Step 8: End-to-End Evaluation"
fi

# 打印总结
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total:  $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
else
    echo "Failed: $FAILED"
fi
echo "=========================================="

# 返回退出码
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
