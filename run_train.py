import sys
import os
import traceback

os.environ['PYTHONUNBUFFERED'] = '1'

try:
    from training.train_classifier import main
    main()
except BaseException as e:
    print(f"\n>>> CAPTURED EXCEPTION IN RUN_TRAIN: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.exit(1)
