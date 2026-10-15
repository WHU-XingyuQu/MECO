/* Three maintenance families, bounded in-flight tickets, no heap or model inference. */
#include <stdint.h>
#include <stddef.h>
#define REG(a) (*(volatile uint32_t *)(a))
#define USART 0x40004400u
#define GPIOD 0x40020C00u
#ifndef DEVICE
#define DEVICE 1u
#endif
#if DEVICE < 1 || DEVICE > 3
#error DEVICE must be in 1..3
#endif
#define CAPACITY 128u
#define MAX_FRAME 160u
typedef struct { uint32_t seq, context, valid, b, c; int32_t temp; } Ticket;
static Ticket pending[CAPACITY];
static uint32_t session, completed, rejected, local_alarm;
static volatile uint32_t tick_ms;
static volatile uint16_t rx_head, rx_tail;
static volatile uint8_t rx_ring[512], rx_overflow;
static char output[160];
static size_t out_len;
extern uint32_t _stack_top, _data_load, _data_start, _data_end, _bss_start, _bss_end;
void Reset_Handler(void);
void SysTick_Handler(void);
void USART2_Handler(void);
void Default_Handler(void) { for (;;) __asm volatile("wfi"); }
__attribute__((section(".isr_vector"), used))
const uintptr_t vectors[16 + 82] = {
    [0]=(uintptr_t)&_stack_top, [1]=(uintptr_t)Reset_Handler,
    [2]=(uintptr_t)Default_Handler, [3]=(uintptr_t)Default_Handler,
    [4]=(uintptr_t)Default_Handler, [5]=(uintptr_t)Default_Handler,
    [6]=(uintptr_t)Default_Handler, [15]=(uintptr_t)SysTick_Handler,
    [16 + 38]=(uintptr_t)USART2_Handler
};
static int equal(const char *a,const char *b) { while(*a && *a==*b) {++a;++b;} return *a==*b; }
static uint16_t crc16(const char *s,size_t n) {
    uint16_t crc=0xFFFF;
    for(size_t i=0;i<n;i++) { crc^=(uint16_t)(uint8_t)s[i]<<8;
        for(unsigned b=0;b<8;b++) crc=(uint16_t)((crc<<1)^((crc&0x8000)?0x1021:0)); }
    return crc;
}
static void tx(char c) { while(!(REG(USART)&(1u<<7))) {} REG(USART+4)=(uint8_t)c; }
static void send(void) {
    uint16_t crc=crc16(output,out_len); const char *hex="0123456789ABCDEF";
    for(size_t i=0;i<out_len;i++) tx(output[i]); tx('*');
    for(int n=12;n>=0;n-=4) tx(hex[(crc>>n)&15]); tx('\n');
}
static void begin(const char *kind) { out_len=0; while(*kind) output[out_len++]=*kind++; }
static void word(const char *s) { output[out_len++]=','; while(*s) output[out_len++]=*s++; }
static void number(int64_t value) {
    char tmp[12]; unsigned n=0; output[out_len++]=',';
    if(value<0) {output[out_len++]='-';value=-value;}
    do {tmp[n++]=(char)('0'+value%10);value/=10;} while(value);
    while(n) output[out_len++]=tmp[--n];
}
static void header(const char *kind) {begin(kind);number(3);number(DEVICE);number(session);}
static void gpio_update(void) {
    uint32_t bits=((tick_ms/1000)&1u)<<12 | (completed&1u)<<13 | local_alarm<<14 | (rejected&1u)<<15;
    REG(GPIOD+0x18)=bits | ((~bits & 0xF000u)<<16);
}
static void error(const char *message) {if(session) {header("E");word(message);send();}}
static int integer(const char *s,int64_t min,int64_t max,int64_t *out) {
    int negative=0;int64_t n=0;
    if(*s=='-') {negative=1;++s;}
    if(!*s || (*s=='0' && s[1])) return 0;
    while(*s) {if(*s<'0'||*s>'9') return 0;n=n*10+(*s++-'0');if(n>4294967295LL) return 0;}
    if(negative) {if(!n) return 0;n=-n;}
    if(n<min||n>max) return 0;*out=n;return 1;
}
static unsigned diagnosis(const Ticket *t) {
    unsigned variant=(t->context-1)%4;
    if(!t->valid) return 1;
    if(DEVICE==1) {
        if(t->temp>=(int32_t)(750+30*variant) && t->b>=700+100*variant) return 2;
        if(t->c<15+2*variant && t->b>=700+100*variant) return 3;
    } else if(DEVICE==2) {
        if(t->temp>=(int32_t)(650+30*variant)) return 4;
        if(t->b<200+20*variant && t->c>=400+50*variant) return 5;
    } else {
        if(t->temp>=(int32_t)(700+30*variant)) return 6;
        if(t->b>=8000+500*variant && t->c>=200+25*variant) return 7;
    }
    return 0;
}
static void process(char *line,size_t size) {
    if(size<6 || line[size-5]!='*') {error("FRAME");return;}
    size_t payload_size=size-5;uint16_t checksum=0;
    for(size_t i=payload_size+1;i<size;i++) {
        char c=line[i];unsigned v;
        if(c>='0'&&c<='9') v=(unsigned)(c-'0');
        else if(c>='A'&&c<='F') v=(unsigned)(c-'A'+10);
        else {error("CRC");return;}
        checksum=(uint16_t)((checksum<<4)|v);
    }
    if(checksum!=crc16(line,payload_size)) {error("CRC");return;}
    line[payload_size]=0;char *f[12];unsigned count=1;f[0]=line;
    for(size_t i=0;i<payload_size;i++) if(line[i]==',') {
        line[i]=0;if(count==12) {error("FRAME");return;}f[count++]=line+i+1;
    }
    int64_t dev,sid,seq;
    if(count<4 || !equal(f[1],"3") || !integer(f[2],1,3,&dev) ||
       !integer(f[3],1,4294967295LL,&sid) || (uint32_t)dev!=DEVICE) {error("FRAME");return;}
    if(equal(f[0],"H") && count==4) {
        if((uint32_t)sid!=session) {
            session=(uint32_t)sid;completed=rejected=0;
            for(unsigned i=0;i<CAPACITY;i++) pending[i].seq=0;
        }
        header("R");word("READY");number(completed);number(tick_ms);send();return;
    }
    if((uint32_t)sid!=session) {error("SESSION");return;}
    if(count<5 || !integer(f[4],1,4294967295LL,&seq)) {error("FRAME");return;}
    if(equal(f[0],"Q") && count==10) {
        int64_t context,temp,b,c,valid;
        if(!integer(f[5],1,12,&context) || ((uint32_t)context-1)/4+1!=DEVICE ||
           !integer(f[6],-400,1500,&temp) || !integer(f[7],0,65535,&b) ||
           !integer(f[8],0,65535,&c) || !integer(f[9],0,1,&valid)) {error("RANGE");return;}
        Ticket *slot=0;
        for(unsigned i=0;i<CAPACITY;i++) {
            if(pending[i].seq==(uint32_t)seq) {error("DUPLICATE");return;}
            if(!pending[i].seq && !slot) slot=&pending[i];
        }
        if(!slot) {error("CAPACITY");return;}
        *slot=(Ticket){(uint32_t)seq,(uint32_t)context,(uint32_t)valid,(uint32_t)b,(uint32_t)c,(int32_t)temp};
        unsigned thermal=DEVICE==1?750:DEVICE==2?650:700;
        local_alarm=!valid || temp>=(int64_t)(thermal+30*(((unsigned)context-1)%4));
        gpio_update();header("T");number(seq);number(context);number(temp);number(b);number(c);number(valid);
        number(tick_ms);send();return;
    }
    if(equal(f[0],"D") && count==7) {
        int64_t code,action;
        if(!integer(f[5],0,255,&code)||!integer(f[6],0,255,&action)) {error("RANGE");return;}
        Ticket *slot=0;
        for(unsigned i=0;i<CAPACITY;i++) if(pending[i].seq==(uint32_t)seq) {slot=&pending[i];break;}
        if(!slot) {error("TASK");return;}
        unsigned expected=diagnosis(slot);
        int accepted=(unsigned)code==expected && (unsigned)action==expected;
        if(accepted) ++completed;else ++rejected;
        slot->seq=0;gpio_update();header("A");number(seq);number(code);number(action);
        word(accepted?"OK":"REJECTED");number(completed);number(tick_ms);number(REG(GPIOD+0x14));send();return;
    }
    error("FRAME");
}
void SysTick_Handler(void) {++tick_ms;}
void USART2_Handler(void) {
    if(REG(USART)&(1u<<5)) {uint8_t c=(uint8_t)REG(USART+4);uint16_t next=(rx_head+1)&511;
        if(next==rx_tail) rx_overflow=1;else {rx_ring[rx_head]=c;rx_head=next;}}
}
static void main_loop(void) {
    REG(0xE000ED08)=0x08000000;
    REG(0x40023830)|=(1u<<0)|(1u<<3);REG(0x40023840)|=1u<<17;
    REG(0x40020000)=(REG(0x40020000)&~(15u<<4))|(10u<<4);
    REG(0x40020020)=(REG(0x40020020)&~(255u<<8))|(0x77u<<8);
    REG(GPIOD)=(REG(GPIOD)&0x00FFFFFFu)|0x55000000u;
    REG(USART+8)=139;REG(USART+0x0C)=(1u<<13)|(1u<<5)|(1u<<3)|(1u<<2);
    REG(0xE000E104)=1u<<6;REG(0xE000E014)=15999;REG(0xE000E018)=0;REG(0xE000E010)=7;
    __asm volatile("cpsie i");
    char line[MAX_FRAME];size_t used=0;int dropping=0;
    for(;;) {
        if(rx_overflow) {rx_overflow=0;used=0;dropping=1;error("OVERFLOW");}
        while(rx_tail!=rx_head) {
            char c=(char)rx_ring[rx_tail];rx_tail=(rx_tail+1)&511;
            if(dropping) {if(c=='\n') dropping=0;continue;}
            if(c=='\n') {line[used]=0;process(line,used);used=0;}
            else if(used>=MAX_FRAME-2) {used=0;dropping=1;error("OVERFLOW");}
            else line[used++]=c;
        }
        __asm volatile("wfi");
    }
}
void Reset_Handler(void) {
    uint32_t *source=&_data_load;
    for(uint32_t *dest=&_data_start;dest<&_data_end;) *dest++=*source++;
    for(uint32_t *dest=&_bss_start;dest<&_bss_end;) *dest++=0;
    main_loop();
}
