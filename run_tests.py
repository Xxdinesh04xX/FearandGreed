"""
Comprehensive test runner for GoQuant Sentiment Trader.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(command, description, capture_output=True):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
        else:
            result = subprocess.run(command, shell=True, check=True)
        
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Command: {command}")
        if capture_output and e.stderr:
            print(f"  Error: {e.stderr}")
        return False


def run_unit_tests():
    """Run unit tests."""
    print("=" * 50)
    print("RUNNING UNIT TESTS")
    print("=" * 50)
    
    success = run_command(
        "python -m pytest tests/ -v --tb=short -m 'not integration'",
        "Running unit tests"
    )
    return success


def run_integration_tests():
    """Run integration tests."""
    print("=" * 50)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 50)
    
    success = run_command(
        "python -m pytest tests/test_integration.py -v --tb=short -m integration",
        "Running integration tests"
    )
    return success


def run_coverage_tests():
    """Run tests with coverage."""
    print("=" * 50)
    print("RUNNING COVERAGE TESTS")
    print("=" * 50)
    
    success = run_command(
        "python -m pytest tests/ --cov=src/goquant --cov-report=html --cov-report=term-missing",
        "Running tests with coverage"
    )
    
    if success:
        print("\nCoverage report generated in htmlcov/index.html")
    
    return success


def run_linting():
    """Run code linting."""
    print("=" * 50)
    print("RUNNING CODE LINTING")
    print("=" * 50)
    
    # Run black formatting check
    black_success = run_command(
        "python -m black --check src/ tests/",
        "Checking code formatting with black"
    )
    
    # Run flake8 linting
    flake8_success = run_command(
        "python -m flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503",
        "Running flake8 linting"
    )
    
    return black_success and flake8_success


def run_type_checking():
    """Run type checking with mypy."""
    print("=" * 50)
    print("RUNNING TYPE CHECKING")
    print("=" * 50)
    
    success = run_command(
        "python -m mypy src/goquant --ignore-missing-imports",
        "Running mypy type checking"
    )
    return success


def run_security_check():
    """Run security checks."""
    print("=" * 50)
    print("RUNNING SECURITY CHECKS")
    print("=" * 50)
    
    # Install bandit if not present
    try:
        import bandit
    except ImportError:
        print("Installing bandit for security checks...")
        subprocess.run([sys.executable, "-m", "pip", "install", "bandit"], check=True)
    
    success = run_command(
        "python -m bandit -r src/ -f json -o bandit_report.json",
        "Running bandit security checks"
    )
    
    if success:
        print("Security report generated in bandit_report.json")
    
    return success


def run_performance_tests():
    """Run performance tests."""
    print("=" * 50)
    print("RUNNING PERFORMANCE TESTS")
    print("=" * 50)
    
    # Simple performance test
    performance_test = """
import time
import asyncio
from src.goquant.utils.text_processor import TextProcessor
from src.goquant.utils.rate_limiter import RateLimiter

async def test_performance():
    # Test text processing performance
    processor = TextProcessor()
    
    test_texts = [
        "Great day for $AAPL and $TSLA trading!",
        "Bitcoin is going to the moon! $BTC",
        "Market crash incoming, sell everything!",
        "Neutral market conditions today."
    ] * 100  # 400 texts
    
    start_time = time.time()
    for text in test_texts:
        processor.clean_text(text)
        processor.extract_financial_symbols(text)
    end_time = time.time()
    
    processing_time = end_time - start_time
    texts_per_second = len(test_texts) / processing_time
    
    print(f"Processed {len(test_texts)} texts in {processing_time:.2f} seconds")
    print(f"Performance: {texts_per_second:.1f} texts/second")
    
    # Test rate limiter performance
    rate_limiter = RateLimiter()
    rate_limiter.add_limit("test", 100, 1)  # 100 requests per second
    
    start_time = time.time()
    for i in range(50):
        await rate_limiter.acquire("test")
    end_time = time.time()
    
    rate_limit_time = end_time - start_time
    print(f"Rate limiter processed 50 requests in {rate_limit_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(test_performance())
"""
    
    # Write performance test to temporary file
    with open("temp_performance_test.py", "w") as f:
        f.write(performance_test)
    
    try:
        success = run_command(
            "python temp_performance_test.py",
            "Running performance tests"
        )
    finally:
        # Clean up temporary file
        if os.path.exists("temp_performance_test.py"):
            os.remove("temp_performance_test.py")
    
    return success


def run_all_tests():
    """Run all tests and checks."""
    print("GoQuant Sentiment Trader - Comprehensive Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Run all test categories
    results['unit_tests'] = run_unit_tests()
    results['integration_tests'] = run_integration_tests()
    results['coverage'] = run_coverage_tests()
    results['linting'] = run_linting()
    results['type_checking'] = run_type_checking()
    results['security'] = run_security_check()
    results['performance'] = run_performance_tests()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title():<20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} test categories passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please review and fix issues before deployment.")
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="GoQuant Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true", help="Run coverage tests")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    
    args = parser.parse_args()
    
    # If no specific test is requested, run all
    if not any([args.unit, args.integration, args.coverage, args.lint, 
                args.type_check, args.security, args.performance]):
        args.all = True
    
    success = True
    
    if args.all:
        success = run_all_tests()
    else:
        if args.unit:
            success &= run_unit_tests()
        if args.integration:
            success &= run_integration_tests()
        if args.coverage:
            success &= run_coverage_tests()
        if args.lint:
            success &= run_linting()
        if args.type_check:
            success &= run_type_checking()
        if args.security:
            success &= run_security_check()
        if args.performance:
            success &= run_performance_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
