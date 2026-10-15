ENTRY(Reset_Handler)
MEMORY { FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 1M
         RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 128K }
_stack_top = ORIGIN(RAM) + LENGTH(RAM);
SECTIONS {
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
  .text : { *(.text*) *(.rodata*) } > FLASH
  .data : { _data_start = .; *(.data*) _data_end = .; } > RAM AT > FLASH
  _data_load = LOADADDR(.data);
  .bss (NOLOAD) : { _bss_start = .; *(.bss*) *(COMMON) _bss_end = .; } > RAM AT > RAM
  _stack_limit = _stack_top - 4096;
  .stack _stack_limit (NOLOAD) : AT(_stack_limit) { . += 4096; } > RAM
  ASSERT(_bss_end <= _stack_limit, "RAM/stack overflow")
  /DISCARD/ : { *(.comment*) *(.note*) *(.ARM.exidx*) }
}
