# MtG Scan Extractor
This is WIP and largely undocumented. Better options may exist somewhere, but I didn't find them.

## Requirements
Python-3.11 or newer

## Setup

```sh
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

## Run

Put images or `.pdf` files into `in` folder.

```sh
python mtg_scan_extractor.py -i in -o out
```

### Alternatives

1. Center based on frame rather than the card via `-c`
2. Test with faster iteration via downsampling with `-d 4`

## TODO
- Clear border after centering
- Add optional controls for better centering
- Remove debug crap
