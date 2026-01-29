#include "LKH.h"

GainType Penalty_M1_PDTSP(void)
{
    GainType P = Penalty_M_PDTSP();
    return P > CurrentPenalty ? P : P + Penalty_SOP();
}
