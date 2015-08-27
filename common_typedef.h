#ifndef _COMMON_TYPEDEF_H_
#define _COMMON_TYPEDEF_H_

/** \brief Usage: BOOLEAN bool; */
typedef bool BOOLEAN;

/** \brief Usage: UWORD8 u8Tmp; */
typedef unsigned char UWORD8;

/** \brief Usage: WORD8 i8Tmp; */
typedef char WORD8;

/** \brief Usage: UWORD16 u16Tmp; */
typedef unsigned short UWORD16;

/** \brief Usage: WORD16 i16Tmp; */
typedef short WORD16;

/** \brief Usage: UWORD32 u32Tmp; */
typedef unsigned int UWORD32;

/** \brief Usage: WORD32 i32Tmp; */
typedef int WORD32;


/** \brief Usage: WORD64 i64Tmp; */
typedef long WORD64;

/** \brief Usage: UWORD64 u64Tmp; */
typedef unsigned long long UWORD64;

/** \brief Usage: FLOAT32  floatTmp; */
typedef float FLOAT32;

/** \brief Usage: DOUBLE64 doubleTmp; */
typedef double DOUBLE64;

/** \brief Usage: COMPLEX16 cp16Tmp; */
typedef struct {
    WORD16 re;
    WORD16 im;
} COMPLEX16;
/** \value it may take 0,1 or 1,-1*/
typedef	signed short  Bits; 

#endif /* #ifndef _COMMON_TYPEDEF_H_ */
