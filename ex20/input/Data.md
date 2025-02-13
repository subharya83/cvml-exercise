# Preprocessing Data into 4 min clips

## Core rationale

Audio Enhancement:

- Amplify Speech: The volume=2.0 filter increases the volume.
- Suppress Noise: The afftdn=nf=-25 filter reduces ambient noise.
- Resample: The audio is resampled to 16KHz mono using -ar 16000 -ac 1.

## Usage

```shell
./preprocess.sh Rangabelia\ Rajapur_\ Event\ area\ 24.07.24.mp3 R
./preprocess.sh Satjelia\ Mitrabari_Event\ area_27.06.24.mp3 S
```