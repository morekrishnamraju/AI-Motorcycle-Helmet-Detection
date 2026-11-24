"""
AI-Based Motorcycle Helmet Detection and Rider Safety Monitoring System
Self-Test Verification Script
"""

import sys
from pathlib import Path

# Add current workspace to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def run_diagnostics():
    print("=" * 70)
    print("  RUNNING SYSTEM DIAGNOSTICS & VERIFICATION")
    print("=" * 70)

    # 1. Test Config Module
    try:
        from modules.config import CLASSES, CLASS_COLORS_BGR, DEFAULT_CONF_THRESHOLD, SAMPLES_DIR
        print(f"[✓] modules.config: OK (Classes: {list(CLASSES.values())})")
    except Exception as e:
        print(f"[✗] modules.config ERROR: {e}")
        return False

    # 2. Test Utils & Sample Generation
    try:
        from modules.utils import generate_synthetic_samples
        generate_synthetic_samples()
        sample_count = len(list(SAMPLES_DIR.glob("*.*")))
        print(f"[✓] modules.utils: OK ({sample_count} sample files verified)")
    except Exception as e:
        print(f"[✗] modules.utils ERROR: {e}")
        return False

    # 3. Test Tracker
    try:
        from modules.tracker import SimpleTracker
        tracker = SimpleTracker()
        dummy_dets = [
            {"bbox": [100, 100, 200, 200], "class": "Helmet", "confidence": 0.88, "is_violation": False},
            {"bbox": [250, 150, 350, 250], "class": "Without Helmet", "confidence": 0.92, "is_violation": True}
        ]
        tracked = tracker.update(dummy_dets)
        print(f"[✓] modules.tracker: OK (Assigned {len(tracked)} persistent Track IDs)")
    except Exception as e:
        print(f"[✗] modules.tracker ERROR: {e}")
        return False

    # 4. Test Analytics
    try:
        from modules.analytics import SafetyAnalytics
        analytics = SafetyAnalytics()
        analytics.log_detections(dummy_dets, source_type="test", source_name="diagnostic_run")
        df = analytics.load_history()
        fig_pie = analytics.create_compliance_donut(helmets=5, without_helmets=2)
        fig_bar = analytics.create_class_bar_chart({"helmets": 5, "without_helmets": 2, "motorcycles": 4, "riders": 7})
        print(f"[✓] modules.analytics: OK (Plotly visualizations & CSV logger verified)")
    except Exception as e:
        print(f"[✗] modules.analytics ERROR: {e}")
        return False

    # 5. Test Report Generator
    try:
        from modules.report_generator import ReportGenerator
        csv_path, _ = ReportGenerator.generate_csv_report(dummy_dets, "Test_Diagnostic")
        pdf_path, _ = ReportGenerator.generate_pdf_report(
            dummy_dets,
            {"total_detections": 2, "helmets": 1, "without_helmets": 1, "compliance_rate": 50.0},
            "Test_Diagnostic"
        )
        print(f"[✓] modules.report_generator: OK (Generated {csv_path.name} and {pdf_path.name})")
    except Exception as e:
        print(f"[✗] modules.report_generator ERROR: {e}")
        return False

    # 6. Test Detector
    try:
        from modules.detector import HelmetDetector
        detector = HelmetDetector()
        print(f"[✓] modules.detector: OK (Mode: {detector.model_mode})")
    except Exception as e:
        print(f"[✗] modules.detector ERROR: {e}")
        return False

    print("=" * 70)
    print("  ALL SYSTEM MODULES VERIFIED SUCCESSFULLY!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
