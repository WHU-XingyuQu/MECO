#!/usr/bin/env bash
set -euo pipefail
course_root="$(cd "$(dirname "$0")" && pwd)"
out="${1:?Pass a new firmware output directory}"
device="${2:?Pass device 1, 2 or 3}"
case "$device" in 1|2|3) ;; *) exit 2 ;; esac
mkdir -p "$out"
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=soft -Os -g3 -std=c11 -DDEVICE="$device" \
  -ffreestanding -fno-builtin -fstack-usage -ffunction-sections -fdata-sections -Wall -Wextra -Werror \
  -Wno-misleading-indentation -nostdlib -Wl,--gc-sections -Wl,-Map="$out/device.map" \
  -T "$course_root/firmware/stm32f407.ld" "$course_root/firmware/main.c" -lgcc -o "$out/device.elf"
arm-none-eabi-size "$out/device.elf" > "$out/size.txt"
arm-none-eabi-readelf -l "$out/device.elf" > "$out/elf-segments.txt"
sha256sum "$out/device.elf" "$course_root/firmware/main.c" > "$out/sha256.txt"
