import os

# Fast handcrafted scorer by default; tests of the CNN select it explicitly.
os.environ.setdefault("REASSEMBLE_SCORER", "handcrafted")
