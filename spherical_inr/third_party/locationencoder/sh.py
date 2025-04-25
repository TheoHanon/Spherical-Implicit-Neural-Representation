# =============================================================================
# Auto-generated real spherical harmonics (ℓ ≤ 30)
# From: https://github.com/marccoru/locationencoder  (MIT-licensed)
# Generated on: 2025-04-24
# =============================================================================

import torch
from torch import cos, sin

def SH(l: int, m: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    try:
        fn = _SH_DISPATCH[(l, m)]

    except KeyError:
        raise ValueError(f"No SH for (l={l},m={m})")
    
    return fn(theta, phi)


@torch.jit.script
def Yl0_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return torch.full_like(theta, 0.282094791773878)

@torch.jit.script
def Yl1_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.48860251190292*(1.0 - cos(theta)**2)**0.5*sin(phi)

@torch.jit.script
def Yl1_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.48860251190292*cos(theta)

@torch.jit.script
def Yl1_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.48860251190292*(1.0 - cos(theta)**2)**0.5*cos(phi)

@torch.jit.script
def Yl2_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.18209140509868*(3.0 - 3.0*cos(theta)**2)*sin(2*phi)

@torch.jit.script
def Yl2_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.09254843059208*(1.0 - cos(theta)**2)**0.5*sin(phi)*cos(theta)

@torch.jit.script
def Yl2_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.94617469575756*cos(theta)**2 - 0.31539156525252

@torch.jit.script
def Yl2_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.09254843059208*(1.0 - cos(theta)**2)**0.5*cos(phi)*cos(theta)

@torch.jit.script
def Yl2_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.18209140509868*(3.0 - 3.0*cos(theta)**2)*cos(2*phi)

@torch.jit.script
def Yl3_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.590043589926644*(1.0 - cos(theta)**2)**1.5*sin(3*phi)

@torch.jit.script
def Yl3_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.44530572132028*(1.0 - cos(theta)**2)*sin(2*phi)*cos(theta)

@torch.jit.script
def Yl3_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.304697199642977*(1.0 - cos(theta)**2)**0.5*(7.5*cos(theta)**2 - 1.5)*sin(phi)

@torch.jit.script
def Yl3_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.86588166295058*cos(theta)**3 - 1.11952899777035*cos(theta)

@torch.jit.script
def Yl3_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.304697199642977*(1.0 - cos(theta)**2)**0.5*(7.5*cos(theta)**2 - 1.5)*cos(phi)

@torch.jit.script
def Yl3_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.44530572132028*(1.0 - cos(theta)**2)*cos(2*phi)*cos(theta)

@torch.jit.script
def Yl3_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.590043589926644*(1.0 - cos(theta)**2)**1.5*cos(3*phi)

@torch.jit.script
def Yl4_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.625835735449176*(1.0 - cos(theta)**2)**2*sin(4*phi)

@torch.jit.script
def Yl4_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.77013076977993*(1.0 - cos(theta)**2)**1.5*sin(3*phi)*cos(theta)

@torch.jit.script
def Yl4_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.063078313050504*(1.0 - cos(theta)**2)*(52.5*cos(theta)**2 - 7.5)*sin(2*phi)

@torch.jit.script
def Yl4_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.267618617422916*(1.0 - cos(theta)**2)**0.5*(17.5*cos(theta)**3 - 7.5*cos(theta))*sin(phi)

@torch.jit.script
def Yl4_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.70249414203215*cos(theta)**4 - 3.17356640745613*cos(theta)**2 + 0.317356640745613

@torch.jit.script
def Yl4_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.267618617422916*(1.0 - cos(theta)**2)**0.5*(17.5*cos(theta)**3 - 7.5*cos(theta))*cos(phi)

@torch.jit.script
def Yl4_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.063078313050504*(1.0 - cos(theta)**2)*(52.5*cos(theta)**2 - 7.5)*cos(2*phi)

@torch.jit.script
def Yl4_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.77013076977993*(1.0 - cos(theta)**2)**1.5*cos(3*phi)*cos(theta)

@torch.jit.script
def Yl4_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.625835735449176*(1.0 - cos(theta)**2)**2*cos(4*phi)

@torch.jit.script
def Yl5_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.65638205684017*(1.0 - cos(theta)**2)**2.5*sin(5*phi)

@torch.jit.script
def Yl5_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.07566231488104*(1.0 - cos(theta)**2)**2*sin(4*phi)*cos(theta)

@torch.jit.script
def Yl5_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00931882475114763*(1.0 - cos(theta)**2)**1.5*(472.5*cos(theta)**2 - 52.5)*sin(3*phi)

@torch.jit.script
def Yl5_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0456527312854602*(1.0 - cos(theta)**2)*(157.5*cos(theta)**3 - 52.5*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl5_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.241571547304372*(1.0 - cos(theta)**2)**0.5*(39.375*cos(theta)**4 - 26.25*cos(theta)**2 + 1.875)*sin(phi)

@torch.jit.script
def Yl5_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.36787031456569*cos(theta)**5 - 8.18652257173965*cos(theta)**3 + 1.75425483680135*cos(theta)

@torch.jit.script
def Yl5_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.241571547304372*(1.0 - cos(theta)**2)**0.5*(39.375*cos(theta)**4 - 26.25*cos(theta)**2 + 1.875)*cos(phi)

@torch.jit.script
def Yl5_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0456527312854602*(1.0 - cos(theta)**2)*(157.5*cos(theta)**3 - 52.5*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl5_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00931882475114763*(1.0 - cos(theta)**2)**1.5*(472.5*cos(theta)**2 - 52.5)*cos(3*phi)

@torch.jit.script
def Yl5_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.07566231488104*(1.0 - cos(theta)**2)**2*cos(4*phi)*cos(theta)

@torch.jit.script
def Yl5_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.65638205684017*(1.0 - cos(theta)**2)**2.5*cos(5*phi)

@torch.jit.script
def Yl6_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.683184105191914*(1.0 - cos(theta)**2)**3*sin(6*phi)

@torch.jit.script
def Yl6_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.36661916223175*(1.0 - cos(theta)**2)**2.5*sin(5*phi)*cos(theta)

@torch.jit.script
def Yl6_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0010678622237645*(1.0 - cos(theta)**2)**2*(5197.5*cos(theta)**2 - 472.5)*sin(4*phi)

@torch.jit.script
def Yl6_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00584892228263444*(1.0 - cos(theta)**2)**1.5*(1732.5*cos(theta)**3 - 472.5*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl6_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0350935336958066*(1.0 - cos(theta)**2)*(433.125*cos(theta)**4 - 236.25*cos(theta)**2 + 13.125)*sin(2*phi)

@torch.jit.script
def Yl6_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.221950995245231*(1.0 - cos(theta)**2)**0.5*(86.625*cos(theta)**5 - 78.75*cos(theta)**3 + 13.125*cos(theta))*sin(phi)

@torch.jit.script
def Yl6_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 14.6844857238222*cos(theta)**6 - 20.024298714303*cos(theta)**4 + 6.67476623810098*cos(theta)**2 - 0.317846011338142

@torch.jit.script
def Yl6_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.221950995245231*(1.0 - cos(theta)**2)**0.5*(86.625*cos(theta)**5 - 78.75*cos(theta)**3 + 13.125*cos(theta))*cos(phi)

@torch.jit.script
def Yl6_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0350935336958066*(1.0 - cos(theta)**2)*(433.125*cos(theta)**4 - 236.25*cos(theta)**2 + 13.125)*cos(2*phi)

@torch.jit.script
def Yl6_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00584892228263444*(1.0 - cos(theta)**2)**1.5*(1732.5*cos(theta)**3 - 472.5*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl6_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0010678622237645*(1.0 - cos(theta)**2)**2*(5197.5*cos(theta)**2 - 472.5)*cos(4*phi)

@torch.jit.script
def Yl6_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.36661916223175*(1.0 - cos(theta)**2)**2.5*cos(5*phi)*cos(theta)

@torch.jit.script
def Yl6_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.683184105191914*(1.0 - cos(theta)**2)**3*cos(6*phi)

@torch.jit.script
def Yl7_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.707162732524596*(1.0 - cos(theta)**2)**3.5*sin(7*phi)

@torch.jit.script
def Yl7_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.6459606618019*(1.0 - cos(theta)**2)**3*sin(6*phi)*cos(theta)

@torch.jit.script
def Yl7_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.98394571852353e-5*(1.0 - cos(theta)**2)**2.5*(67567.5*cos(theta)**2 - 5197.5)*sin(5*phi)

@torch.jit.script
def Yl7_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000599036743111412*(1.0 - cos(theta)**2)**2*(22522.5*cos(theta)**3 - 5197.5*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl7_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00397356022507413*(1.0 - cos(theta)**2)**1.5*(5630.625*cos(theta)**4 - 2598.75*cos(theta)**2 + 118.125)*sin(3*phi)

@torch.jit.script
def Yl7_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0280973138060306*(1.0 - cos(theta)**2)*(1126.125*cos(theta)**5 - 866.25*cos(theta)**3 + 118.125*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl7_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.206472245902897*(1.0 - cos(theta)**2)**0.5*(187.6875*cos(theta)**6 - 216.5625*cos(theta)**4 + 59.0625*cos(theta)**2 - 2.1875)*sin(phi)

@torch.jit.script
def Yl7_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 29.2939547952501*cos(theta)**7 - 47.3210039000194*cos(theta)**5 + 21.5095472272816*cos(theta)**3 - 2.38994969192017*cos(theta)

@torch.jit.script
def Yl7_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.206472245902897*(1.0 - cos(theta)**2)**0.5*(187.6875*cos(theta)**6 - 216.5625*cos(theta)**4 + 59.0625*cos(theta)**2 - 2.1875)*cos(phi)

@torch.jit.script
def Yl7_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0280973138060306*(1.0 - cos(theta)**2)*(1126.125*cos(theta)**5 - 866.25*cos(theta)**3 + 118.125*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl7_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00397356022507413*(1.0 - cos(theta)**2)**1.5*(5630.625*cos(theta)**4 - 2598.75*cos(theta)**2 + 118.125)*cos(3*phi)

@torch.jit.script
def Yl7_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000599036743111412*(1.0 - cos(theta)**2)**2*(22522.5*cos(theta)**3 - 5197.5*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl7_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.98394571852353e-5*(1.0 - cos(theta)**2)**2.5*(67567.5*cos(theta)**2 - 5197.5)*cos(5*phi)

@torch.jit.script
def Yl7_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.6459606618019*(1.0 - cos(theta)**2)**3*cos(6*phi)*cos(theta)

@torch.jit.script
def Yl7_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.707162732524596*(1.0 - cos(theta)**2)**3.5*cos(7*phi)

@torch.jit.script
def Yl8_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.72892666017483*(1.0 - cos(theta)**2)**4*sin(8*phi)

@torch.jit.script
def Yl8_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.91570664069932*(1.0 - cos(theta)**2)**3.5*sin(7*phi)*cos(theta)

@torch.jit.script
def Yl8_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.87853281621404e-6*(1.0 - cos(theta)**2)**3*(1013512.5*cos(theta)**2 - 67567.5)*sin(6*phi)

@torch.jit.script
def Yl8_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.10587282657803e-5*(1.0 - cos(theta)**2)**2.5*(337837.5*cos(theta)**3 - 67567.5*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl8_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000368189725644507*(1.0 - cos(theta)**2)**2*(84459.375*cos(theta)**4 - 33783.75*cos(theta)**2 + 1299.375)*sin(4*phi)

@torch.jit.script
def Yl8_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0028519853513317*(1.0 - cos(theta)**2)**1.5*(16891.875*cos(theta)**5 - 11261.25*cos(theta)**3 + 1299.375*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl8_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0231696385236779*(1.0 - cos(theta)**2)*(2815.3125*cos(theta)**6 - 2815.3125*cos(theta)**4 + 649.6875*cos(theta)**2 - 19.6875)*sin(2*phi)

@torch.jit.script
def Yl8_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.193851103820053*(1.0 - cos(theta)**2)**0.5*(402.1875*cos(theta)**7 - 563.0625*cos(theta)**5 + 216.5625*cos(theta)**3 - 19.6875*cos(theta))*sin(phi)

@torch.jit.script
def Yl8_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 58.4733681132208*cos(theta)**8 - 109.150287144679*cos(theta)**6 + 62.9713195065454*cos(theta)**4 - 11.4493308193719*cos(theta)**2 + 0.318036967204775

@torch.jit.script
def Yl8_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.193851103820053*(1.0 - cos(theta)**2)**0.5*(402.1875*cos(theta)**7 - 563.0625*cos(theta)**5 + 216.5625*cos(theta)**3 - 19.6875*cos(theta))*cos(phi)

@torch.jit.script
def Yl8_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0231696385236779*(1.0 - cos(theta)**2)*(2815.3125*cos(theta)**6 - 2815.3125*cos(theta)**4 + 649.6875*cos(theta)**2 - 19.6875)*cos(2*phi)

@torch.jit.script
def Yl8_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0028519853513317*(1.0 - cos(theta)**2)**1.5*(16891.875*cos(theta)**5 - 11261.25*cos(theta)**3 + 1299.375*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl8_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000368189725644507*(1.0 - cos(theta)**2)**2*(84459.375*cos(theta)**4 - 33783.75*cos(theta)**2 + 1299.375)*cos(4*phi)

@torch.jit.script
def Yl8_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.10587282657803e-5*(1.0 - cos(theta)**2)**2.5*(337837.5*cos(theta)**3 - 67567.5*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl8_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.87853281621404e-6*(1.0 - cos(theta)**2)**3*(1013512.5*cos(theta)**2 - 67567.5)*cos(6*phi)

@torch.jit.script
def Yl8_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.91570664069932*(1.0 - cos(theta)**2)**3.5*cos(7*phi)*cos(theta)

@torch.jit.script
def Yl8_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.72892666017483*(1.0 - cos(theta)**2)**4*cos(8*phi)

@torch.jit.script
def Yl9_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.748900951853188*(1.0 - cos(theta)**2)**4.5*sin(9*phi)

@torch.jit.script
def Yl9_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.1773176489547*(1.0 - cos(theta)**2)**4*sin(8*phi)*cos(theta)

@torch.jit.script
def Yl9_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.37640612566745e-7*(1.0 - cos(theta)**2)**3.5*(17229712.5*cos(theta)**2 - 1013512.5)*sin(7*phi)

@torch.jit.script
def Yl9_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.72488342871223e-6*(1.0 - cos(theta)**2)**3*(5743237.5*cos(theta)**3 - 1013512.5*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl9_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.88528229719329e-5*(1.0 - cos(theta)**2)**2.5*(1435809.375*cos(theta)**4 - 506756.25*cos(theta)**2 + 16891.875)*sin(5*phi)

@torch.jit.script
def Yl9_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000241400036332803*(1.0 - cos(theta)**2)**2*(287161.875*cos(theta)**5 - 168918.75*cos(theta)**3 + 16891.875*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl9_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00213198739401417*(1.0 - cos(theta)**2)**1.5*(47860.3125*cos(theta)**6 - 42229.6875*cos(theta)**4 + 8445.9375*cos(theta)**2 - 216.5625)*sin(3*phi)

@torch.jit.script
def Yl9_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0195399872275232*(1.0 - cos(theta)**2)*(6837.1875*cos(theta)**7 - 8445.9375*cos(theta)**5 + 2815.3125*cos(theta)**3 - 216.5625*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl9_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.183301328077446*(1.0 - cos(theta)**2)**0.5*(854.6484375*cos(theta)**8 - 1407.65625*cos(theta)**6 + 703.828125*cos(theta)**4 - 108.28125*cos(theta)**2 + 2.4609375)*sin(phi)

@torch.jit.script
def Yl9_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 116.766123398619*cos(theta)**9 - 247.269437785311*cos(theta)**7 + 173.088606449718*cos(theta)**5 - 44.3816939614661*cos(theta)**3 + 3.02602458828178*cos(theta)

@torch.jit.script
def Yl9_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.183301328077446*(1.0 - cos(theta)**2)**0.5*(854.6484375*cos(theta)**8 - 1407.65625*cos(theta)**6 + 703.828125*cos(theta)**4 - 108.28125*cos(theta)**2 + 2.4609375)*cos(phi)

@torch.jit.script
def Yl9_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0195399872275232*(1.0 - cos(theta)**2)*(6837.1875*cos(theta)**7 - 8445.9375*cos(theta)**5 + 2815.3125*cos(theta)**3 - 216.5625*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl9_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00213198739401417*(1.0 - cos(theta)**2)**1.5*(47860.3125*cos(theta)**6 - 42229.6875*cos(theta)**4 + 8445.9375*cos(theta)**2 - 216.5625)*cos(3*phi)

@torch.jit.script
def Yl9_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000241400036332803*(1.0 - cos(theta)**2)**2*(287161.875*cos(theta)**5 - 168918.75*cos(theta)**3 + 16891.875*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl9_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.88528229719329e-5*(1.0 - cos(theta)**2)**2.5*(1435809.375*cos(theta)**4 - 506756.25*cos(theta)**2 + 16891.875)*cos(5*phi)

@torch.jit.script
def Yl9_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.72488342871223e-6*(1.0 - cos(theta)**2)**3*(5743237.5*cos(theta)**3 - 1013512.5*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl9_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.37640612566745e-7*(1.0 - cos(theta)**2)**3.5*(17229712.5*cos(theta)**2 - 1013512.5)*cos(7*phi)

@torch.jit.script
def Yl9_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.1773176489547*(1.0 - cos(theta)**2)**4*cos(8*phi)*cos(theta)

@torch.jit.script
def Yl9_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.748900951853188*(1.0 - cos(theta)**2)**4.5*cos(9*phi)

@torch.jit.script
def Yl10_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.76739511822199*(1.0 - cos(theta)**2)**5*sin(10*phi)

@torch.jit.script
def Yl10_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.43189529989171*(1.0 - cos(theta)**2)**4.5*sin(9*phi)*cos(theta)

@torch.jit.script
def Yl10_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.23120268385452e-8*(1.0 - cos(theta)**2)**4*(327364537.5*cos(theta)**2 - 17229712.5)*sin(8*phi)

@torch.jit.script
def Yl10_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.37443934928654e-7*(1.0 - cos(theta)**2)**3.5*(109121512.5*cos(theta)**3 - 17229712.5*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl10_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.95801284774625e-6*(1.0 - cos(theta)**2)**3*(27280378.125*cos(theta)**4 - 8614856.25*cos(theta)**2 + 253378.125)*sin(6*phi)

@torch.jit.script
def Yl10_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.75129993135143e-5*(1.0 - cos(theta)**2)**2.5*(5456075.625*cos(theta)**5 - 2871618.75*cos(theta)**3 + 253378.125*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl10_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000166142899475011*(1.0 - cos(theta)**2)**2*(909345.9375*cos(theta)**6 - 717904.6875*cos(theta)**4 + 126689.0625*cos(theta)**2 - 2815.3125)*sin(4*phi)

@torch.jit.script
def Yl10_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00164473079210685*(1.0 - cos(theta)**2)**1.5*(129906.5625*cos(theta)**7 - 143580.9375*cos(theta)**5 + 42229.6875*cos(theta)**3 - 2815.3125*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl10_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0167730288071195*(1.0 - cos(theta)**2)*(16238.3203125*cos(theta)**8 - 23930.15625*cos(theta)**6 + 10557.421875*cos(theta)**4 - 1407.65625*cos(theta)**2 + 27.0703125)*sin(2*phi)

@torch.jit.script
def Yl10_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.174310428544485*(1.0 - cos(theta)**2)**0.5*(1804.2578125*cos(theta)**9 - 3418.59375*cos(theta)**7 + 2111.484375*cos(theta)**5 - 469.21875*cos(theta)**3 + 27.0703125*cos(theta))*sin(phi)

@torch.jit.script
def Yl10_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 233.240148813258*cos(theta)**10 - 552.410878768242*cos(theta)**8 + 454.926606044435*cos(theta)**6 - 151.642202014812*cos(theta)**4 + 17.4971771555552*cos(theta)**2 - 0.318130493737367

@torch.jit.script
def Yl10_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.174310428544485*(1.0 - cos(theta)**2)**0.5*(1804.2578125*cos(theta)**9 - 3418.59375*cos(theta)**7 + 2111.484375*cos(theta)**5 - 469.21875*cos(theta)**3 + 27.0703125*cos(theta))*cos(phi)

@torch.jit.script
def Yl10_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0167730288071195*(1.0 - cos(theta)**2)*(16238.3203125*cos(theta)**8 - 23930.15625*cos(theta)**6 + 10557.421875*cos(theta)**4 - 1407.65625*cos(theta)**2 + 27.0703125)*cos(2*phi)

@torch.jit.script
def Yl10_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00164473079210685*(1.0 - cos(theta)**2)**1.5*(129906.5625*cos(theta)**7 - 143580.9375*cos(theta)**5 + 42229.6875*cos(theta)**3 - 2815.3125*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl10_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000166142899475011*(1.0 - cos(theta)**2)**2*(909345.9375*cos(theta)**6 - 717904.6875*cos(theta)**4 + 126689.0625*cos(theta)**2 - 2815.3125)*cos(4*phi)

@torch.jit.script
def Yl10_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.75129993135143e-5*(1.0 - cos(theta)**2)**2.5*(5456075.625*cos(theta)**5 - 2871618.75*cos(theta)**3 + 253378.125*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl10_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.95801284774625e-6*(1.0 - cos(theta)**2)**3*(27280378.125*cos(theta)**4 - 8614856.25*cos(theta)**2 + 253378.125)*cos(6*phi)

@torch.jit.script
def Yl10_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.37443934928654e-7*(1.0 - cos(theta)**2)**3.5*(109121512.5*cos(theta)**3 - 17229712.5*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl10_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.23120268385452e-8*(1.0 - cos(theta)**2)**4*(327364537.5*cos(theta)**2 - 17229712.5)*cos(8*phi)

@torch.jit.script
def Yl10_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.43189529989171*(1.0 - cos(theta)**2)**4.5*cos(9*phi)*cos(theta)

@torch.jit.script
def Yl10_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.76739511822199*(1.0 - cos(theta)**2)**5*cos(10*phi)

@torch.jit.script
def Yl11_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.784642105787197*(1.0 - cos(theta)**2)**5.5*sin(11*phi)

@torch.jit.script
def Yl11_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.68029769880531*(1.0 - cos(theta)**2)**5*sin(10*phi)*cos(theta)

@torch.jit.script
def Yl11_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.73470916587426e-9*(1.0 - cos(theta)**2)**4.5*(6874655287.5*cos(theta)**2 - 327364537.5)*sin(9*phi)

@torch.jit.script
def Yl11_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.34369994198887e-8*(1.0 - cos(theta)**2)**4*(2291551762.5*cos(theta)**3 - 327364537.5*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl11_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.17141045151419e-7*(1.0 - cos(theta)**2)**3.5*(572887940.625*cos(theta)**4 - 163682268.75*cos(theta)**2 + 4307428.125)*sin(7*phi)

@torch.jit.script
def Yl11_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.11129753051333e-6*(1.0 - cos(theta)**2)**3*(114577588.125*cos(theta)**5 - 54560756.25*cos(theta)**3 + 4307428.125*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl11_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.12235548974089e-5*(1.0 - cos(theta)**2)**2.5*(19096264.6875*cos(theta)**6 - 13640189.0625*cos(theta)**4 + 2153714.0625*cos(theta)**2 - 42229.6875)*sin(5*phi)

@torch.jit.script
def Yl11_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0001187789403385*(1.0 - cos(theta)**2)**2*(2728037.8125*cos(theta)**7 - 2728037.8125*cos(theta)**5 + 717904.6875*cos(theta)**3 - 42229.6875*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl11_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00130115809959914*(1.0 - cos(theta)**2)**1.5*(341004.7265625*cos(theta)**8 - 454672.96875*cos(theta)**6 + 179476.171875*cos(theta)**4 - 21114.84375*cos(theta)**2 + 351.9140625)*sin(3*phi)

@torch.jit.script
def Yl11_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0146054634441776*(1.0 - cos(theta)**2)*(37889.4140625*cos(theta)**9 - 64953.28125*cos(theta)**7 + 35895.234375*cos(theta)**5 - 7038.28125*cos(theta)**3 + 351.9140625*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl11_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.166527904912351*(1.0 - cos(theta)**2)**0.5*(3788.94140625*cos(theta)**10 - 8119.16015625*cos(theta)**8 + 5982.5390625*cos(theta)**6 - 1759.5703125*cos(theta)**4 + 175.95703125*cos(theta)**2 - 2.70703125)*sin(phi)

@torch.jit.script
def Yl11_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 465.998147319252*cos(theta)**11 - 1220.47133821709*cos(theta)**9 + 1156.23600462672*cos(theta)**7 - 476.097178375706*cos(theta)**5 + 79.3495297292844*cos(theta)**3 - 3.66228598750543*cos(theta)

@torch.jit.script
def Yl11_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.166527904912351*(1.0 - cos(theta)**2)**0.5*(3788.94140625*cos(theta)**10 - 8119.16015625*cos(theta)**8 + 5982.5390625*cos(theta)**6 - 1759.5703125*cos(theta)**4 + 175.95703125*cos(theta)**2 - 2.70703125)*cos(phi)

@torch.jit.script
def Yl11_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0146054634441776*(1.0 - cos(theta)**2)*(37889.4140625*cos(theta)**9 - 64953.28125*cos(theta)**7 + 35895.234375*cos(theta)**5 - 7038.28125*cos(theta)**3 + 351.9140625*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl11_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00130115809959914*(1.0 - cos(theta)**2)**1.5*(341004.7265625*cos(theta)**8 - 454672.96875*cos(theta)**6 + 179476.171875*cos(theta)**4 - 21114.84375*cos(theta)**2 + 351.9140625)*cos(3*phi)

@torch.jit.script
def Yl11_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0001187789403385*(1.0 - cos(theta)**2)**2*(2728037.8125*cos(theta)**7 - 2728037.8125*cos(theta)**5 + 717904.6875*cos(theta)**3 - 42229.6875*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl11_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.12235548974089e-5*(1.0 - cos(theta)**2)**2.5*(19096264.6875*cos(theta)**6 - 13640189.0625*cos(theta)**4 + 2153714.0625*cos(theta)**2 - 42229.6875)*cos(5*phi)

@torch.jit.script
def Yl11_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.11129753051333e-6*(1.0 - cos(theta)**2)**3*(114577588.125*cos(theta)**5 - 54560756.25*cos(theta)**3 + 4307428.125*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl11_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.17141045151419e-7*(1.0 - cos(theta)**2)**3.5*(572887940.625*cos(theta)**4 - 163682268.75*cos(theta)**2 + 4307428.125)*cos(7*phi)

@torch.jit.script
def Yl11_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.34369994198887e-8*(1.0 - cos(theta)**2)**4*(2291551762.5*cos(theta)**3 - 327364537.5*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl11_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.73470916587426e-9*(1.0 - cos(theta)**2)**4.5*(6874655287.5*cos(theta)**2 - 327364537.5)*cos(9*phi)

@torch.jit.script
def Yl11_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.68029769880531*(1.0 - cos(theta)**2)**5*cos(10*phi)*cos(theta)

@torch.jit.script
def Yl11_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.784642105787197*(1.0 - cos(theta)**2)**5.5*cos(11*phi)

@torch.jit.script
def Yl12_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.800821995783972*(1.0 - cos(theta)**2)**6*sin(12*phi)

@torch.jit.script
def Yl12_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.92321052893599*(1.0 - cos(theta)**2)**5.5*sin(11*phi)*cos(theta)

@torch.jit.script
def Yl12_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.4141794839602e-11*(1.0 - cos(theta)**2)**5*(158117071612.5*cos(theta)**2 - 6874655287.5)*sin(10*phi)

@torch.jit.script
def Yl12_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.83571172711927e-10*(1.0 - cos(theta)**2)**4.5*(52705690537.5*cos(theta)**3 - 6874655287.5*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl12_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.26503328368427e-9*(1.0 - cos(theta)**2)**4*(13176422634.375*cos(theta)**4 - 3437327643.75*cos(theta)**2 + 81841134.375)*sin(8*phi)

@torch.jit.script
def Yl12_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.26503328368427e-8*(1.0 - cos(theta)**2)**3.5*(2635284526.875*cos(theta)**5 - 1145775881.25*cos(theta)**3 + 81841134.375*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl12_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.68922506214776e-7*(1.0 - cos(theta)**2)**3*(439214087.8125*cos(theta)**6 - 286443970.3125*cos(theta)**4 + 40920567.1875*cos(theta)**2 - 717904.6875)*sin(6*phi)

@torch.jit.script
def Yl12_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.50863650967357e-6*(1.0 - cos(theta)**2)**2.5*(62744869.6875*cos(theta)**7 - 57288794.0625*cos(theta)**5 + 13640189.0625*cos(theta)**3 - 717904.6875*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl12_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.75649965675714e-5*(1.0 - cos(theta)**2)**2*(7843108.7109375*cos(theta)**8 - 9548132.34375*cos(theta)**6 + 3410047.265625*cos(theta)**4 - 358952.34375*cos(theta)**2 + 5278.7109375)*sin(4*phi)

@torch.jit.script
def Yl12_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00105077995881086*(1.0 - cos(theta)**2)**1.5*(871456.5234375*cos(theta)**9 - 1364018.90625*cos(theta)**7 + 682009.453125*cos(theta)**5 - 119650.78125*cos(theta)**3 + 5278.7109375*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl12_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0128693736551466*(1.0 - cos(theta)**2)*(87145.65234375*cos(theta)**10 - 170502.36328125*cos(theta)**8 + 113668.2421875*cos(theta)**6 - 29912.6953125*cos(theta)**4 + 2639.35546875*cos(theta)**2 - 35.19140625)*sin(2*phi)

@torch.jit.script
def Yl12_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.159704727088682*(1.0 - cos(theta)**2)**0.5*(7922.33203125*cos(theta)**11 - 18944.70703125*cos(theta)**9 + 16238.3203125*cos(theta)**7 - 5982.5390625*cos(theta)**5 + 879.78515625*cos(theta)**3 - 35.19140625*cos(theta))*sin(phi)

@torch.jit.script
def Yl12_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 931.186918632914*cos(theta)**12 - 2672.1015925988*cos(theta)**10 + 2862.96599207014*cos(theta)**8 - 1406.36925926252*cos(theta)**6 + 310.228513072616*cos(theta)**4 - 24.8182810458093*cos(theta)**2 + 0.318183090330888

@torch.jit.script
def Yl12_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.159704727088682*(1.0 - cos(theta)**2)**0.5*(7922.33203125*cos(theta)**11 - 18944.70703125*cos(theta)**9 + 16238.3203125*cos(theta)**7 - 5982.5390625*cos(theta)**5 + 879.78515625*cos(theta)**3 - 35.19140625*cos(theta))*cos(phi)

@torch.jit.script
def Yl12_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0128693736551466*(1.0 - cos(theta)**2)*(87145.65234375*cos(theta)**10 - 170502.36328125*cos(theta)**8 + 113668.2421875*cos(theta)**6 - 29912.6953125*cos(theta)**4 + 2639.35546875*cos(theta)**2 - 35.19140625)*cos(2*phi)

@torch.jit.script
def Yl12_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00105077995881086*(1.0 - cos(theta)**2)**1.5*(871456.5234375*cos(theta)**9 - 1364018.90625*cos(theta)**7 + 682009.453125*cos(theta)**5 - 119650.78125*cos(theta)**3 + 5278.7109375*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl12_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.75649965675714e-5*(1.0 - cos(theta)**2)**2*(7843108.7109375*cos(theta)**8 - 9548132.34375*cos(theta)**6 + 3410047.265625*cos(theta)**4 - 358952.34375*cos(theta)**2 + 5278.7109375)*cos(4*phi)

@torch.jit.script
def Yl12_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.50863650967357e-6*(1.0 - cos(theta)**2)**2.5*(62744869.6875*cos(theta)**7 - 57288794.0625*cos(theta)**5 + 13640189.0625*cos(theta)**3 - 717904.6875*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl12_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.68922506214776e-7*(1.0 - cos(theta)**2)**3*(439214087.8125*cos(theta)**6 - 286443970.3125*cos(theta)**4 + 40920567.1875*cos(theta)**2 - 717904.6875)*cos(6*phi)

@torch.jit.script
def Yl12_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.26503328368427e-8*(1.0 - cos(theta)**2)**3.5*(2635284526.875*cos(theta)**5 - 1145775881.25*cos(theta)**3 + 81841134.375*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl12_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.26503328368427e-9*(1.0 - cos(theta)**2)**4*(13176422634.375*cos(theta)**4 - 3437327643.75*cos(theta)**2 + 81841134.375)*cos(8*phi)

@torch.jit.script
def Yl12_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.83571172711927e-10*(1.0 - cos(theta)**2)**4.5*(52705690537.5*cos(theta)**3 - 6874655287.5*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl12_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.4141794839602e-11*(1.0 - cos(theta)**2)**5*(158117071612.5*cos(theta)**2 - 6874655287.5)*cos(10*phi)

@torch.jit.script
def Yl12_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.92321052893598*(1.0 - cos(theta)**2)**5.5*cos(11*phi)*cos(theta)

@torch.jit.script
def Yl12_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.800821995783972*(1.0 - cos(theta)**2)**6*cos(12*phi)

@torch.jit.script
def Yl13_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.816077118837628*(1.0 - cos(theta)**2)**6.5*sin(13*phi)

@torch.jit.script
def Yl13_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.16119315354964*(1.0 - cos(theta)**2)**6*sin(12*phi)*cos(theta)

@torch.jit.script
def Yl13_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.72180924766049e-12*(1.0 - cos(theta)**2)**5.5*(3952926790312.5*cos(theta)**2 - 158117071612.5)*sin(11*phi)

@torch.jit.script
def Yl13_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.15805986876424e-11*(1.0 - cos(theta)**2)**5*(1317642263437.5*cos(theta)**3 - 158117071612.5*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl13_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.02910461422567e-10*(1.0 - cos(theta)**2)**4.5*(329410565859.375*cos(theta)**4 - 79058535806.25*cos(theta)**2 + 1718663821.875)*sin(9*phi)

@torch.jit.script
def Yl13_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.17695172143292e-9*(1.0 - cos(theta)**2)**4*(65882113171.875*cos(theta)**5 - 26352845268.75*cos(theta)**3 + 1718663821.875*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl13_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.5661194627771e-8*(1.0 - cos(theta)**2)**3.5*(10980352195.3125*cos(theta)**6 - 6588211317.1875*cos(theta)**4 + 859331910.9375*cos(theta)**2 - 13640189.0625)*sin(7*phi)

@torch.jit.script
def Yl13_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.21948945157073e-7*(1.0 - cos(theta)**2)**3*(1568621742.1875*cos(theta)**7 - 1317642263.4375*cos(theta)**5 + 286443970.3125*cos(theta)**3 - 13640189.0625*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl13_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.2021359721285e-6*(1.0 - cos(theta)**2)**2.5*(196077717.773438*cos(theta)**8 - 219607043.90625*cos(theta)**6 + 71610992.578125*cos(theta)**4 - 6820094.53125*cos(theta)**2 + 89738.0859375)*sin(5*phi)

@torch.jit.script
def Yl13_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.62123812058377e-5*(1.0 - cos(theta)**2)**2*(21786413.0859375*cos(theta)**9 - 31372434.84375*cos(theta)**7 + 14322198.515625*cos(theta)**5 - 2273364.84375*cos(theta)**3 + 89738.0859375*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl13_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000863303829622583*(1.0 - cos(theta)**2)**1.5*(2178641.30859375*cos(theta)**10 - 3921554.35546875*cos(theta)**8 + 2387033.0859375*cos(theta)**6 - 568341.2109375*cos(theta)**4 + 44869.04296875*cos(theta)**2 - 527.87109375)*sin(3*phi)

@torch.jit.script
def Yl13_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0114530195317401*(1.0 - cos(theta)**2)*(198058.30078125*cos(theta)**11 - 435728.26171875*cos(theta)**9 + 341004.7265625*cos(theta)**7 - 113668.2421875*cos(theta)**5 + 14956.34765625*cos(theta)**3 - 527.87109375*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl13_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.153658381323621*(1.0 - cos(theta)**2)**0.5*(16504.8583984375*cos(theta)**12 - 43572.826171875*cos(theta)**10 + 42625.5908203125*cos(theta)**8 - 18944.70703125*cos(theta)**6 + 3739.0869140625*cos(theta)**4 - 263.935546875*cos(theta)**2 + 2.9326171875)*sin(phi)

@torch.jit.script
def Yl13_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1860.99583201813*cos(theta)**13 - 5806.30699589657*cos(theta)**11 + 6942.32358205025*cos(theta)**9 - 3967.04204688585*cos(theta)**7 + 1096.15635506056*cos(theta)**5 - 128.959571183596*cos(theta)**3 + 4.29865237278653*cos(theta)

@torch.jit.script
def Yl13_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.153658381323621*(1.0 - cos(theta)**2)**0.5*(16504.8583984375*cos(theta)**12 - 43572.826171875*cos(theta)**10 + 42625.5908203125*cos(theta)**8 - 18944.70703125*cos(theta)**6 + 3739.0869140625*cos(theta)**4 - 263.935546875*cos(theta)**2 + 2.9326171875)*cos(phi)

@torch.jit.script
def Yl13_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0114530195317401*(1.0 - cos(theta)**2)*(198058.30078125*cos(theta)**11 - 435728.26171875*cos(theta)**9 + 341004.7265625*cos(theta)**7 - 113668.2421875*cos(theta)**5 + 14956.34765625*cos(theta)**3 - 527.87109375*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl13_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000863303829622583*(1.0 - cos(theta)**2)**1.5*(2178641.30859375*cos(theta)**10 - 3921554.35546875*cos(theta)**8 + 2387033.0859375*cos(theta)**6 - 568341.2109375*cos(theta)**4 + 44869.04296875*cos(theta)**2 - 527.87109375)*cos(3*phi)

@torch.jit.script
def Yl13_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.62123812058377e-5*(1.0 - cos(theta)**2)**2*(21786413.0859375*cos(theta)**9 - 31372434.84375*cos(theta)**7 + 14322198.515625*cos(theta)**5 - 2273364.84375*cos(theta)**3 + 89738.0859375*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl13_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.2021359721285e-6*(1.0 - cos(theta)**2)**2.5*(196077717.773438*cos(theta)**8 - 219607043.90625*cos(theta)**6 + 71610992.578125*cos(theta)**4 - 6820094.53125*cos(theta)**2 + 89738.0859375)*cos(5*phi)

@torch.jit.script
def Yl13_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.21948945157073e-7*(1.0 - cos(theta)**2)**3*(1568621742.1875*cos(theta)**7 - 1317642263.4375*cos(theta)**5 + 286443970.3125*cos(theta)**3 - 13640189.0625*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl13_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.5661194627771e-8*(1.0 - cos(theta)**2)**3.5*(10980352195.3125*cos(theta)**6 - 6588211317.1875*cos(theta)**4 + 859331910.9375*cos(theta)**2 - 13640189.0625)*cos(7*phi)

@torch.jit.script
def Yl13_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.17695172143292e-9*(1.0 - cos(theta)**2)**4*(65882113171.875*cos(theta)**5 - 26352845268.75*cos(theta)**3 + 1718663821.875*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl13_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.02910461422567e-10*(1.0 - cos(theta)**2)**4.5*(329410565859.375*cos(theta)**4 - 79058535806.25*cos(theta)**2 + 1718663821.875)*cos(9*phi)

@torch.jit.script
def Yl13_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.15805986876424e-11*(1.0 - cos(theta)**2)**5*(1317642263437.5*cos(theta)**3 - 158117071612.5*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl13_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.72180924766049e-12*(1.0 - cos(theta)**2)**5.5*(3952926790312.5*cos(theta)**2 - 158117071612.5)*cos(11*phi)

@torch.jit.script
def Yl13_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.16119315354964*(1.0 - cos(theta)**2)**6*cos(12*phi)*cos(theta)

@torch.jit.script
def Yl13_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.816077118837628*(1.0 - cos(theta)**2)**6.5*cos(13*phi)

@torch.jit.script
def Yl14_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.830522083064524*(1.0 - cos(theta)**2)**7*sin(14*phi)

@torch.jit.script
def Yl14_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.39470978027212*(1.0 - cos(theta)**2)**6.5*sin(13*phi)*cos(theta)

@torch.jit.script
def Yl14_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.51291507116349e-13*(1.0 - cos(theta)**2)**6*(106729023338438.0*cos(theta)**2 - 3952926790312.5)*sin(12*phi)

@torch.jit.script
def Yl14_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.33617041195793e-12*(1.0 - cos(theta)**2)**5.5*(35576341112812.5*cos(theta)**3 - 3952926790312.5*cos(theta))*sin(11*phi)

@torch.jit.script
def Yl14_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.33617041195793e-11*(1.0 - cos(theta)**2)**5*(8894085278203.13*cos(theta)**4 - 1976463395156.25*cos(theta)**2 + 39529267903.125)*sin(10*phi)

@torch.jit.script
def Yl14_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.46370135060066e-10*(1.0 - cos(theta)**2)**4.5*(1778817055640.63*cos(theta)**5 - 658821131718.75*cos(theta)**3 + 39529267903.125*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl14_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.71945976061531e-9*(1.0 - cos(theta)**2)**4*(296469509273.438*cos(theta)**6 - 164705282929.688*cos(theta)**4 + 19764633951.5625*cos(theta)**2 - 286443970.3125)*sin(8*phi)

@torch.jit.script
def Yl14_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.13379344766496e-8*(1.0 - cos(theta)**2)**3.5*(42352787039.0625*cos(theta)**7 - 32941056585.9375*cos(theta)**5 + 6588211317.1875*cos(theta)**3 - 286443970.3125*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl14_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.76571240765567e-7*(1.0 - cos(theta)**2)**3*(5294098379.88281*cos(theta)**8 - 5490176097.65625*cos(theta)**6 + 1647052829.29688*cos(theta)**4 - 143221985.15625*cos(theta)**2 + 1705023.6328125)*sin(6*phi)

@torch.jit.script
def Yl14_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.71059256983961e-6*(1.0 - cos(theta)**2)**2.5*(588233153.320313*cos(theta)**9 - 784310871.09375*cos(theta)**7 + 329410565.859375*cos(theta)**5 - 47740661.71875*cos(theta)**3 + 1705023.6328125*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl14_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.11469888818129e-5*(1.0 - cos(theta)**2)**2*(58823315.3320313*cos(theta)**10 - 98038858.8867188*cos(theta)**8 + 54901760.9765625*cos(theta)**6 - 11935165.4296875*cos(theta)**4 + 852511.81640625*cos(theta)**2 - 8973.80859375)*sin(4*phi)

@torch.jit.script
def Yl14_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000719701928156307*(1.0 - cos(theta)**2)**1.5*(5347574.12109375*cos(theta)**11 - 10893206.5429688*cos(theta)**9 + 7843108.7109375*cos(theta)**7 - 2387033.0859375*cos(theta)**5 + 284170.60546875*cos(theta)**3 - 8973.80859375*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl14_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0102793996196251*(1.0 - cos(theta)**2)*(445631.176757813*cos(theta)**12 - 1089320.65429688*cos(theta)**10 + 980388.588867188*cos(theta)**8 - 397838.84765625*cos(theta)**6 + 71042.6513671875*cos(theta)**4 - 4486.904296875*cos(theta)**2 + 43.9892578125)*sin(2*phi)

@torch.jit.script
def Yl14_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.148251609638173*(1.0 - cos(theta)**2)**0.5*(34279.3212890625*cos(theta)**13 - 99029.150390625*cos(theta)**11 + 108932.065429688*cos(theta)**9 - 56834.12109375*cos(theta)**7 + 14208.5302734375*cos(theta)**5 - 1495.634765625*cos(theta)**3 + 43.9892578125*cos(theta))*sin(phi)

@torch.jit.script
def Yl14_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3719.61718745389*cos(theta)**14 - 12536.487557715*cos(theta)**12 + 16548.1635761838*cos(theta)**10 - 10792.2805931633*cos(theta)**8 + 3597.42686438778*cos(theta)**6 - 568.014768061228*cos(theta)**4 + 33.4126334153663*cos(theta)**2 - 0.318215556336822

@torch.jit.script
def Yl14_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.148251609638173*(1.0 - cos(theta)**2)**0.5*(34279.3212890625*cos(theta)**13 - 99029.150390625*cos(theta)**11 + 108932.065429688*cos(theta)**9 - 56834.12109375*cos(theta)**7 + 14208.5302734375*cos(theta)**5 - 1495.634765625*cos(theta)**3 + 43.9892578125*cos(theta))*cos(phi)

@torch.jit.script
def Yl14_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.0102793996196251*(1.0 - cos(theta)**2)*(445631.176757813*cos(theta)**12 - 1089320.65429688*cos(theta)**10 + 980388.588867188*cos(theta)**8 - 397838.84765625*cos(theta)**6 + 71042.6513671875*cos(theta)**4 - 4486.904296875*cos(theta)**2 + 43.9892578125)*cos(2*phi)

@torch.jit.script
def Yl14_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000719701928156307*(1.0 - cos(theta)**2)**1.5*(5347574.12109375*cos(theta)**11 - 10893206.5429688*cos(theta)**9 + 7843108.7109375*cos(theta)**7 - 2387033.0859375*cos(theta)**5 + 284170.60546875*cos(theta)**3 - 8973.80859375*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl14_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.11469888818129e-5*(1.0 - cos(theta)**2)**2*(58823315.3320313*cos(theta)**10 - 98038858.8867188*cos(theta)**8 + 54901760.9765625*cos(theta)**6 - 11935165.4296875*cos(theta)**4 + 852511.81640625*cos(theta)**2 - 8973.80859375)*cos(4*phi)

@torch.jit.script
def Yl14_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.71059256983961e-6*(1.0 - cos(theta)**2)**2.5*(588233153.320313*cos(theta)**9 - 784310871.09375*cos(theta)**7 + 329410565.859375*cos(theta)**5 - 47740661.71875*cos(theta)**3 + 1705023.6328125*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl14_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.76571240765567e-7*(1.0 - cos(theta)**2)**3*(5294098379.88281*cos(theta)**8 - 5490176097.65625*cos(theta)**6 + 1647052829.29688*cos(theta)**4 - 143221985.15625*cos(theta)**2 + 1705023.6328125)*cos(6*phi)

@torch.jit.script
def Yl14_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.13379344766496e-8*(1.0 - cos(theta)**2)**3.5*(42352787039.0625*cos(theta)**7 - 32941056585.9375*cos(theta)**5 + 6588211317.1875*cos(theta)**3 - 286443970.3125*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl14_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.71945976061531e-9*(1.0 - cos(theta)**2)**4*(296469509273.438*cos(theta)**6 - 164705282929.688*cos(theta)**4 + 19764633951.5625*cos(theta)**2 - 286443970.3125)*cos(8*phi)

@torch.jit.script
def Yl14_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.46370135060066e-10*(1.0 - cos(theta)**2)**4.5*(1778817055640.63*cos(theta)**5 - 658821131718.75*cos(theta)**3 + 39529267903.125*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl14_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.33617041195793e-11*(1.0 - cos(theta)**2)**5*(8894085278203.13*cos(theta)**4 - 1976463395156.25*cos(theta)**2 + 39529267903.125)*cos(10*phi)

@torch.jit.script
def Yl14_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.33617041195793e-12*(1.0 - cos(theta)**2)**5.5*(35576341112812.5*cos(theta)**3 - 3952926790312.5*cos(theta))*cos(11*phi)

@torch.jit.script
def Yl14_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.51291507116349e-13*(1.0 - cos(theta)**2)**6*(106729023338438.0*cos(theta)**2 - 3952926790312.5)*cos(12*phi)

@torch.jit.script
def Yl14_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.39470978027212*(1.0 - cos(theta)**2)**6.5*cos(13*phi)*cos(theta)

@torch.jit.script
def Yl14_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.830522083064524*(1.0 - cos(theta)**2)**7*cos(14*phi)

@torch.jit.script
def Yl15_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.844250650857373*(1.0 - cos(theta)**2)**7.5*sin(15*phi)

@torch.jit.script
def Yl15_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.62415125663001*(1.0 - cos(theta)**2)**7*sin(14*phi)*cos(theta)

@torch.jit.script
def Yl15_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.68899431025918e-15*(1.0 - cos(theta)**2)**6.5*(3.09514167681469e+15*cos(theta)**2 - 106729023338438.0)*sin(13*phi)

@torch.jit.script
def Yl15_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.21404941098716e-14*(1.0 - cos(theta)**2)**6*(1.03171389227156e+15*cos(theta)**3 - 106729023338438.0*cos(theta))*sin(12*phi)

@torch.jit.script
def Yl15_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.4185990958026e-13*(1.0 - cos(theta)**2)**5.5*(257928473067891.0*cos(theta)**4 - 53364511669218.8*cos(theta)**2 + 988231697578.125)*sin(11*phi)

@torch.jit.script
def Yl15_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.17815352749854e-12*(1.0 - cos(theta)**2)**5*(51585694613578.1*cos(theta)**5 - 17788170556406.3*cos(theta)**3 + 988231697578.125*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl15_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.56666184747369e-11*(1.0 - cos(theta)**2)**4.5*(8597615768929.69*cos(theta)**6 - 4447042639101.56*cos(theta)**4 + 494115848789.063*cos(theta)**2 - 6588211317.1875)*sin(9*phi)

@torch.jit.script
def Yl15_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.80751467720255e-10*(1.0 - cos(theta)**2)**4*(1228230824132.81*cos(theta)**7 - 889408527820.313*cos(theta)**5 + 164705282929.688*cos(theta)**3 - 6588211317.1875*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl15_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.33035601710264e-8*(1.0 - cos(theta)**2)**3.5*(153528853016.602*cos(theta)**8 - 148234754636.719*cos(theta)**6 + 41176320732.4219*cos(theta)**4 - 3294105658.59375*cos(theta)**2 + 35805496.2890625)*sin(7*phi)

@torch.jit.script
def Yl15_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.87197684863824e-7*(1.0 - cos(theta)**2)**3*(17058761446.2891*cos(theta)**9 - 21176393519.5313*cos(theta)**7 + 8235264146.48438*cos(theta)**5 - 1098035219.53125*cos(theta)**3 + 35805496.2890625*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl15_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.71275217737612e-6*(1.0 - cos(theta)**2)**2.5*(1705876144.62891*cos(theta)**10 - 2647049189.94141*cos(theta)**8 + 1372544024.41406*cos(theta)**6 - 274508804.882813*cos(theta)**4 + 17902748.1445313*cos(theta)**2 - 170502.36328125)*sin(5*phi)

@torch.jit.script
def Yl15_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.02366171874445e-5*(1.0 - cos(theta)**2)**2*(155079649.511719*cos(theta)**11 - 294116576.660156*cos(theta)**9 + 196077717.773438*cos(theta)**7 - 54901760.9765625*cos(theta)**5 + 5967582.71484375*cos(theta)**3 - 170502.36328125*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl15_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000607559596001151*(1.0 - cos(theta)**2)**1.5*(12923304.1259766*cos(theta)**12 - 29411657.6660156*cos(theta)**10 + 24509714.7216797*cos(theta)**8 - 9150293.49609375*cos(theta)**6 + 1491895.67871094*cos(theta)**4 - 85251.181640625*cos(theta)**2 + 747.8173828125)*sin(3*phi)

@torch.jit.script
def Yl15_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00929387470704126*(1.0 - cos(theta)**2)*(994100.317382813*cos(theta)**13 - 2673787.06054688*cos(theta)**11 + 2723301.63574219*cos(theta)**9 - 1307184.78515625*cos(theta)**7 + 298379.135742188*cos(theta)**5 - 28417.060546875*cos(theta)**3 + 747.8173828125*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl15_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.143378915753688*(1.0 - cos(theta)**2)**0.5*(71007.1655273438*cos(theta)**14 - 222815.588378906*cos(theta)**12 + 272330.163574219*cos(theta)**10 - 163398.098144531*cos(theta)**8 + 49729.8559570313*cos(theta)**6 - 7104.26513671875*cos(theta)**4 + 373.90869140625*cos(theta)**2 - 3.14208984375)*sin(phi)

@torch.jit.script
def Yl15_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7435.10031825349*cos(theta)**15 - 26920.1908074695*cos(theta)**13 + 38884.7200552338*cos(theta)**11 - 28515.4613738381*cos(theta)**9 + 11158.2240158497*cos(theta)**7 - 2231.64480316994*cos(theta)**5 + 195.758316067539*cos(theta)**3 - 4.93508359834131*cos(theta)

@torch.jit.script
def Yl15_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.143378915753688*(1.0 - cos(theta)**2)**0.5*(71007.1655273438*cos(theta)**14 - 222815.588378906*cos(theta)**12 + 272330.163574219*cos(theta)**10 - 163398.098144531*cos(theta)**8 + 49729.8559570313*cos(theta)**6 - 7104.26513671875*cos(theta)**4 + 373.90869140625*cos(theta)**2 - 3.14208984375)*cos(phi)

@torch.jit.script
def Yl15_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00929387470704126*(1.0 - cos(theta)**2)*(994100.317382813*cos(theta)**13 - 2673787.06054688*cos(theta)**11 + 2723301.63574219*cos(theta)**9 - 1307184.78515625*cos(theta)**7 + 298379.135742188*cos(theta)**5 - 28417.060546875*cos(theta)**3 + 747.8173828125*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl15_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000607559596001151*(1.0 - cos(theta)**2)**1.5*(12923304.1259766*cos(theta)**12 - 29411657.6660156*cos(theta)**10 + 24509714.7216797*cos(theta)**8 - 9150293.49609375*cos(theta)**6 + 1491895.67871094*cos(theta)**4 - 85251.181640625*cos(theta)**2 + 747.8173828125)*cos(3*phi)

@torch.jit.script
def Yl15_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.02366171874445e-5*(1.0 - cos(theta)**2)**2*(155079649.511719*cos(theta)**11 - 294116576.660156*cos(theta)**9 + 196077717.773438*cos(theta)**7 - 54901760.9765625*cos(theta)**5 + 5967582.71484375*cos(theta)**3 - 170502.36328125*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl15_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.71275217737612e-6*(1.0 - cos(theta)**2)**2.5*(1705876144.62891*cos(theta)**10 - 2647049189.94141*cos(theta)**8 + 1372544024.41406*cos(theta)**6 - 274508804.882813*cos(theta)**4 + 17902748.1445313*cos(theta)**2 - 170502.36328125)*cos(5*phi)

@torch.jit.script
def Yl15_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.87197684863824e-7*(1.0 - cos(theta)**2)**3*(17058761446.2891*cos(theta)**9 - 21176393519.5313*cos(theta)**7 + 8235264146.48438*cos(theta)**5 - 1098035219.53125*cos(theta)**3 + 35805496.2890625*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl15_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.33035601710264e-8*(1.0 - cos(theta)**2)**3.5*(153528853016.602*cos(theta)**8 - 148234754636.719*cos(theta)**6 + 41176320732.4219*cos(theta)**4 - 3294105658.59375*cos(theta)**2 + 35805496.2890625)*cos(7*phi)

@torch.jit.script
def Yl15_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.80751467720255e-10*(1.0 - cos(theta)**2)**4*(1228230824132.81*cos(theta)**7 - 889408527820.313*cos(theta)**5 + 164705282929.688*cos(theta)**3 - 6588211317.1875*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl15_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.56666184747369e-11*(1.0 - cos(theta)**2)**4.5*(8597615768929.69*cos(theta)**6 - 4447042639101.56*cos(theta)**4 + 494115848789.063*cos(theta)**2 - 6588211317.1875)*cos(9*phi)

@torch.jit.script
def Yl15_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.17815352749854e-12*(1.0 - cos(theta)**2)**5*(51585694613578.1*cos(theta)**5 - 17788170556406.3*cos(theta)**3 + 988231697578.125*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl15_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.4185990958026e-13*(1.0 - cos(theta)**2)**5.5*(257928473067891.0*cos(theta)**4 - 53364511669218.8*cos(theta)**2 + 988231697578.125)*cos(11*phi)

@torch.jit.script
def Yl15_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.21404941098716e-14*(1.0 - cos(theta)**2)**6*(1.03171389227156e+15*cos(theta)**3 - 106729023338438.0*cos(theta))*cos(12*phi)

@torch.jit.script
def Yl15_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.68899431025918e-15*(1.0 - cos(theta)**2)**6.5*(3.09514167681469e+15*cos(theta)**2 - 106729023338438.0)*cos(13*phi)

@torch.jit.script
def Yl15_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.62415125663001*(1.0 - cos(theta)**2)**7*cos(14*phi)*cos(theta)

@torch.jit.script
def Yl15_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.844250650857373*(1.0 - cos(theta)**2)**7.5*cos(15*phi)

@torch.jit.script
def Yl16_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.857340588838025*(1.0 - cos(theta)**2)**8*sin(16*phi)

@torch.jit.script
def Yl16_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.84985075323068*(1.0 - cos(theta)**2)**7.5*sin(15*phi)*cos(theta)

@torch.jit.script
def Yl16_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.98999505000411e-16*(1.0 - cos(theta)**2)**7*(9.59493919812553e+16*cos(theta)**2 - 3.09514167681469e+15)*sin(14*phi)

@torch.jit.script
def Yl16_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.8878750671421e-15*(1.0 - cos(theta)**2)**6.5*(3.19831306604184e+16*cos(theta)**3 - 3.09514167681469e+15*cos(theta))*sin(13*phi)

@torch.jit.script
def Yl16_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.03330367436807e-14*(1.0 - cos(theta)**2)**6*(7.99578266510461e+15*cos(theta)**4 - 1.54757083840734e+15*cos(theta)**2 + 26682255834609.4)*sin(12*phi)

@torch.jit.script
def Yl16_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.40583735216622e-13*(1.0 - cos(theta)**2)**5.5*(1.59915653302092e+15*cos(theta)**5 - 515856946135781.0*cos(theta)**3 + 26682255834609.4*cos(theta))*sin(11*phi)

@torch.jit.script
def Yl16_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.06213103106751e-12*(1.0 - cos(theta)**2)**5*(266526088836820.0*cos(theta)**6 - 128964236533945.0*cos(theta)**4 + 13341127917304.7*cos(theta)**2 - 164705282929.688)*sin(10*phi)

@torch.jit.script
def Yl16_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.1310406124361e-11*(1.0 - cos(theta)**2)**4.5*(38075155548117.2*cos(theta)**7 - 25792847306789.1*cos(theta)**5 + 4447042639101.56*cos(theta)**3 - 164705282929.688*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl16_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.84217366082119e-10*(1.0 - cos(theta)**2)**4*(4759394443514.65*cos(theta)**8 - 4298807884464.84*cos(theta)**6 + 1111760659775.39*cos(theta)**4 - 82352641464.8438*cos(theta)**2 + 823526414.648438)*sin(8*phi)

@torch.jit.script
def Yl16_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.58620667464373e-9*(1.0 - cos(theta)**2)**3.5*(528821604834.961*cos(theta)**9 - 614115412066.406*cos(theta)**7 + 222352131955.078*cos(theta)**5 - 27450880488.2813*cos(theta)**3 + 823526414.648438*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl16_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.30216271501415e-7*(1.0 - cos(theta)**2)**3*(52882160483.4961*cos(theta)**10 - 76764426508.3008*cos(theta)**8 + 37058688659.1797*cos(theta)**6 - 6862720122.07031*cos(theta)**4 + 411763207.324219*cos(theta)**2 - 3580549.62890625)*sin(6*phi)

@torch.jit.script
def Yl16_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.02568978918854e-6*(1.0 - cos(theta)**2)**2.5*(4807469134.86328*cos(theta)**11 - 8529380723.14453*cos(theta)**9 + 5294098379.88281*cos(theta)**7 - 1372544024.41406*cos(theta)**5 + 137254402.441406*cos(theta)**3 - 3580549.62890625*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl16_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.21568284933344e-5*(1.0 - cos(theta)**2)**2*(400622427.905273*cos(theta)**12 - 852938072.314453*cos(theta)**10 + 661762297.485352*cos(theta)**8 - 228757337.402344*cos(theta)**6 + 34313600.6103516*cos(theta)**4 - 1790274.81445313*cos(theta)**2 + 14208.5302734375)*sin(4*phi)

@torch.jit.script
def Yl16_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000518513279362185*(1.0 - cos(theta)**2)**1.5*(30817109.8388672*cos(theta)**13 - 77539824.7558594*cos(theta)**11 + 73529144.1650391*cos(theta)**9 - 32679619.6289063*cos(theta)**7 + 6862720.12207031*cos(theta)**5 - 596758.271484375*cos(theta)**3 + 14208.5302734375*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl16_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00845669566395355*(1.0 - cos(theta)**2)*(2201222.13134766*cos(theta)**14 - 6461652.06298828*cos(theta)**12 + 7352914.41650391*cos(theta)**10 - 4084952.45361328*cos(theta)**8 + 1143786.68701172*cos(theta)**6 - 149189.567871094*cos(theta)**4 + 7104.26513671875*cos(theta)**2 - 53.41552734375)*sin(2*phi)

@torch.jit.script
def Yl16_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.138957689313105*(1.0 - cos(theta)**2)**0.5*(146748.142089844*cos(theta)**15 - 497050.158691406*cos(theta)**13 + 668446.765136719*cos(theta)**11 - 453883.605957031*cos(theta)**9 + 163398.098144531*cos(theta)**7 - 29837.9135742188*cos(theta)**5 + 2368.08837890625*cos(theta)**3 - 53.41552734375*cos(theta))*sin(phi)

@torch.jit.script
def Yl16_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 14862.9380228203*cos(theta)**16 - 57533.9536367237*cos(theta)**14 + 90268.7893265838*cos(theta)**12 - 73552.3468586979*cos(theta)**10 + 33098.5560864141*cos(theta)**8 - 8058.77887321386*cos(theta)**6 + 959.378437287364*cos(theta)**4 - 43.2802302535653*cos(theta)**2 + 0.318236987158568

@torch.jit.script
def Yl16_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.138957689313105*(1.0 - cos(theta)**2)**0.5*(146748.142089844*cos(theta)**15 - 497050.158691406*cos(theta)**13 + 668446.765136719*cos(theta)**11 - 453883.605957031*cos(theta)**9 + 163398.098144531*cos(theta)**7 - 29837.9135742188*cos(theta)**5 + 2368.08837890625*cos(theta)**3 - 53.41552734375*cos(theta))*cos(phi)

@torch.jit.script
def Yl16_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00845669566395355*(1.0 - cos(theta)**2)*(2201222.13134766*cos(theta)**14 - 6461652.06298828*cos(theta)**12 + 7352914.41650391*cos(theta)**10 - 4084952.45361328*cos(theta)**8 + 1143786.68701172*cos(theta)**6 - 149189.567871094*cos(theta)**4 + 7104.26513671875*cos(theta)**2 - 53.41552734375)*cos(2*phi)

@torch.jit.script
def Yl16_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000518513279362185*(1.0 - cos(theta)**2)**1.5*(30817109.8388672*cos(theta)**13 - 77539824.7558594*cos(theta)**11 + 73529144.1650391*cos(theta)**9 - 32679619.6289063*cos(theta)**7 + 6862720.12207031*cos(theta)**5 - 596758.271484375*cos(theta)**3 + 14208.5302734375*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl16_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.21568284933344e-5*(1.0 - cos(theta)**2)**2*(400622427.905273*cos(theta)**12 - 852938072.314453*cos(theta)**10 + 661762297.485352*cos(theta)**8 - 228757337.402344*cos(theta)**6 + 34313600.6103516*cos(theta)**4 - 1790274.81445313*cos(theta)**2 + 14208.5302734375)*cos(4*phi)

@torch.jit.script
def Yl16_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.02568978918854e-6*(1.0 - cos(theta)**2)**2.5*(4807469134.86328*cos(theta)**11 - 8529380723.14453*cos(theta)**9 + 5294098379.88281*cos(theta)**7 - 1372544024.41406*cos(theta)**5 + 137254402.441406*cos(theta)**3 - 3580549.62890625*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl16_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.30216271501415e-7*(1.0 - cos(theta)**2)**3*(52882160483.4961*cos(theta)**10 - 76764426508.3008*cos(theta)**8 + 37058688659.1797*cos(theta)**6 - 6862720122.07031*cos(theta)**4 + 411763207.324219*cos(theta)**2 - 3580549.62890625)*cos(6*phi)

@torch.jit.script
def Yl16_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.58620667464373e-9*(1.0 - cos(theta)**2)**3.5*(528821604834.961*cos(theta)**9 - 614115412066.406*cos(theta)**7 + 222352131955.078*cos(theta)**5 - 27450880488.2813*cos(theta)**3 + 823526414.648438*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl16_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.84217366082119e-10*(1.0 - cos(theta)**2)**4*(4759394443514.65*cos(theta)**8 - 4298807884464.84*cos(theta)**6 + 1111760659775.39*cos(theta)**4 - 82352641464.8438*cos(theta)**2 + 823526414.648438)*cos(8*phi)

@torch.jit.script
def Yl16_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.1310406124361e-11*(1.0 - cos(theta)**2)**4.5*(38075155548117.2*cos(theta)**7 - 25792847306789.1*cos(theta)**5 + 4447042639101.56*cos(theta)**3 - 164705282929.688*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl16_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.06213103106751e-12*(1.0 - cos(theta)**2)**5*(266526088836820.0*cos(theta)**6 - 128964236533945.0*cos(theta)**4 + 13341127917304.7*cos(theta)**2 - 164705282929.688)*cos(10*phi)

@torch.jit.script
def Yl16_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.40583735216622e-13*(1.0 - cos(theta)**2)**5.5*(1.59915653302092e+15*cos(theta)**5 - 515856946135781.0*cos(theta)**3 + 26682255834609.4*cos(theta))*cos(11*phi)

@torch.jit.script
def Yl16_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.03330367436807e-14*(1.0 - cos(theta)**2)**6*(7.99578266510461e+15*cos(theta)**4 - 1.54757083840734e+15*cos(theta)**2 + 26682255834609.4)*cos(12*phi)

@torch.jit.script
def Yl16_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.8878750671421e-15*(1.0 - cos(theta)**2)**6.5*(3.19831306604184e+16*cos(theta)**3 - 3.09514167681469e+15*cos(theta))*cos(13*phi)

@torch.jit.script
def Yl16_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.98999505000411e-16*(1.0 - cos(theta)**2)**7*(9.59493919812553e+16*cos(theta)**2 - 3.09514167681469e+15)*cos(14*phi)

@torch.jit.script
def Yl16_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.84985075323068*(1.0 - cos(theta)**2)**7.5*cos(15*phi)*cos(theta)

@torch.jit.script
def Yl16_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.857340588838025*(1.0 - cos(theta)**2)**8*cos(16*phi)

@torch.jit.script
def Yl17_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.869857171920628*(1.0 - cos(theta)**2)**8.5*sin(17*phi)

@torch.jit.script
def Yl17_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.07209532485536*(1.0 - cos(theta)**2)**8*sin(16*phi)*cos(theta)

@torch.jit.script
def Yl17_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.50688621401289e-18*(1.0 - cos(theta)**2)**7.5*(3.16632993538143e+18*cos(theta)**2 - 9.59493919812553e+16)*sin(15*phi)

@torch.jit.script
def Yl17_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.37542041547274e-17*(1.0 - cos(theta)**2)**7*(1.05544331179381e+18*cos(theta)**3 - 9.59493919812553e+16*cos(theta))*sin(14*phi)

@torch.jit.script
def Yl17_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.09936771746562e-16*(1.0 - cos(theta)**2)**6.5*(2.63860827948452e+17*cos(theta)**4 - 4.79746959906277e+16*cos(theta)**2 + 773785419203672.0)*sin(13*phi)

@torch.jit.script
def Yl17_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.69491420208903e-15*(1.0 - cos(theta)**2)**6*(5.27721655896904e+16*cos(theta)**5 - 1.59915653302092e+16*cos(theta)**3 + 773785419203672.0*cos(theta))*sin(12*phi)

@torch.jit.script
def Yl17_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.14693795555008e-13*(1.0 - cos(theta)**2)**5.5*(8.79536093161507e+15*cos(theta)**6 - 3.9978913325523e+15*cos(theta)**4 + 386892709601836.0*cos(theta)**2 - 4447042639101.56)*sin(11*phi)

@torch.jit.script
def Yl17_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.60571313777011e-12*(1.0 - cos(theta)**2)**5*(1.25648013308787e+15*cos(theta)**7 - 799578266510461.0*cos(theta)**5 + 128964236533945.0*cos(theta)**3 - 4447042639101.56*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl17_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.35990671649205e-11*(1.0 - cos(theta)**2)**4.5*(157060016635983.0*cos(theta)**8 - 133263044418410.0*cos(theta)**6 + 32241059133486.3*cos(theta)**4 - 2223521319550.78*cos(theta)**2 + 20588160366.2109)*sin(9*phi)

@torch.jit.script
def Yl17_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.60996311929549e-10*(1.0 - cos(theta)**2)**4*(17451112959553.7*cos(theta)**9 - 19037577774058.6*cos(theta)**7 + 6448211826697.27*cos(theta)**5 - 741173773183.594*cos(theta)**3 + 20588160366.2109*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl17_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.70785286308994e-9*(1.0 - cos(theta)**2)**3.5*(1745111295955.37*cos(theta)**10 - 2379697221757.32*cos(theta)**8 + 1074701971116.21*cos(theta)**6 - 185293443295.898*cos(theta)**4 + 10294080183.1055*cos(theta)**2 - 82352641.4648438)*sin(7*phi)

@torch.jit.script
def Yl17_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.2741631735508e-8*(1.0 - cos(theta)**2)**3*(158646481450.488*cos(theta)**11 - 264410802417.48*cos(theta)**9 + 153528853016.602*cos(theta)**7 - 37058688659.1797*cos(theta)**5 + 3431360061.03516*cos(theta)**3 - 82352641.4648438*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl17_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.54073970252026e-6*(1.0 - cos(theta)**2)**2.5*(13220540120.874*cos(theta)**12 - 26441080241.748*cos(theta)**10 + 19191106627.0752*cos(theta)**8 - 6176448109.86328*cos(theta)**6 + 857840015.258789*cos(theta)**4 - 41176320.7324219*cos(theta)**2 + 298379.135742188)*sin(5*phi)

@torch.jit.script
def Yl17_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.6056272673653e-5*(1.0 - cos(theta)**2)**2*(1016964624.68262*cos(theta)**13 - 2403734567.43164*cos(theta)**11 + 2132345180.78613*cos(theta)**9 - 882349729.980469*cos(theta)**7 + 171568003.051758*cos(theta)**5 - 13725440.2441406*cos(theta)**3 + 298379.135742188*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl17_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000446772008544923*(1.0 - cos(theta)**2)**1.5*(72640330.3344727*cos(theta)**14 - 200311213.952637*cos(theta)**12 + 213234518.078613*cos(theta)**10 - 110293716.247559*cos(theta)**8 + 28594667.175293*cos(theta)**6 - 3431360.06103516*cos(theta)**4 + 149189.567871094*cos(theta)**2 - 1014.89501953125)*sin(3*phi)

@torch.jit.script
def Yl17_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00773831818199403*(1.0 - cos(theta)**2)*(4842688.68896484*cos(theta)**15 - 15408554.9194336*cos(theta)**13 + 19384956.1889648*cos(theta)**11 - 12254857.3608398*cos(theta)**9 + 4084952.45361328*cos(theta)**7 - 686272.012207031*cos(theta)**5 + 49729.8559570313*cos(theta)**3 - 1014.89501953125*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl17_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.134922187793101*(1.0 - cos(theta)**2)**0.5*(302668.043060303*cos(theta)**16 - 1100611.06567383*cos(theta)**14 + 1615413.01574707*cos(theta)**12 - 1225485.73608398*cos(theta)**10 + 510619.05670166*cos(theta)**8 - 114378.668701172*cos(theta)**6 + 12432.4639892578*cos(theta)**4 - 507.447509765625*cos(theta)**2 + 3.33847045898438)*sin(phi)

@torch.jit.script
def Yl17_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 29713.0160510757*cos(theta)**17 - 122453.641907463*cos(theta)**15 + 207381.167746511*cos(theta)**13 - 185927.943496872*cos(theta)**11 + 94685.5267808142*cos(theta)**9 - 27269.4317128745*cos(theta)**7 + 4149.69613022003*cos(theta)**5 - 282.292253756465*cos(theta)**3 + 5.57155763993023*cos(theta)

@torch.jit.script
def Yl17_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.134922187793101*(1.0 - cos(theta)**2)**0.5*(302668.043060303*cos(theta)**16 - 1100611.06567383*cos(theta)**14 + 1615413.01574707*cos(theta)**12 - 1225485.73608398*cos(theta)**10 + 510619.05670166*cos(theta)**8 - 114378.668701172*cos(theta)**6 + 12432.4639892578*cos(theta)**4 - 507.447509765625*cos(theta)**2 + 3.33847045898438)*cos(phi)

@torch.jit.script
def Yl17_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00773831818199403*(1.0 - cos(theta)**2)*(4842688.68896484*cos(theta)**15 - 15408554.9194336*cos(theta)**13 + 19384956.1889648*cos(theta)**11 - 12254857.3608398*cos(theta)**9 + 4084952.45361328*cos(theta)**7 - 686272.012207031*cos(theta)**5 + 49729.8559570313*cos(theta)**3 - 1014.89501953125*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl17_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000446772008544923*(1.0 - cos(theta)**2)**1.5*(72640330.3344727*cos(theta)**14 - 200311213.952637*cos(theta)**12 + 213234518.078613*cos(theta)**10 - 110293716.247559*cos(theta)**8 + 28594667.175293*cos(theta)**6 - 3431360.06103516*cos(theta)**4 + 149189.567871094*cos(theta)**2 - 1014.89501953125)*cos(3*phi)

@torch.jit.script
def Yl17_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.6056272673653e-5*(1.0 - cos(theta)**2)**2*(1016964624.68262*cos(theta)**13 - 2403734567.43164*cos(theta)**11 + 2132345180.78613*cos(theta)**9 - 882349729.980469*cos(theta)**7 + 171568003.051758*cos(theta)**5 - 13725440.2441406*cos(theta)**3 + 298379.135742188*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl17_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.54073970252026e-6*(1.0 - cos(theta)**2)**2.5*(13220540120.874*cos(theta)**12 - 26441080241.748*cos(theta)**10 + 19191106627.0752*cos(theta)**8 - 6176448109.86328*cos(theta)**6 + 857840015.258789*cos(theta)**4 - 41176320.7324219*cos(theta)**2 + 298379.135742188)*cos(5*phi)

@torch.jit.script
def Yl17_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.2741631735508e-8*(1.0 - cos(theta)**2)**3*(158646481450.488*cos(theta)**11 - 264410802417.48*cos(theta)**9 + 153528853016.602*cos(theta)**7 - 37058688659.1797*cos(theta)**5 + 3431360061.03516*cos(theta)**3 - 82352641.4648438*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl17_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.70785286308994e-9*(1.0 - cos(theta)**2)**3.5*(1745111295955.37*cos(theta)**10 - 2379697221757.32*cos(theta)**8 + 1074701971116.21*cos(theta)**6 - 185293443295.898*cos(theta)**4 + 10294080183.1055*cos(theta)**2 - 82352641.4648438)*cos(7*phi)

@torch.jit.script
def Yl17_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.60996311929549e-10*(1.0 - cos(theta)**2)**4*(17451112959553.7*cos(theta)**9 - 19037577774058.6*cos(theta)**7 + 6448211826697.27*cos(theta)**5 - 741173773183.594*cos(theta)**3 + 20588160366.2109*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl17_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.35990671649205e-11*(1.0 - cos(theta)**2)**4.5*(157060016635983.0*cos(theta)**8 - 133263044418410.0*cos(theta)**6 + 32241059133486.3*cos(theta)**4 - 2223521319550.78*cos(theta)**2 + 20588160366.2109)*cos(9*phi)

@torch.jit.script
def Yl17_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.60571313777011e-12*(1.0 - cos(theta)**2)**5*(1.25648013308787e+15*cos(theta)**7 - 799578266510461.0*cos(theta)**5 + 128964236533945.0*cos(theta)**3 - 4447042639101.56*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl17_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.14693795555008e-13*(1.0 - cos(theta)**2)**5.5*(8.79536093161507e+15*cos(theta)**6 - 3.9978913325523e+15*cos(theta)**4 + 386892709601836.0*cos(theta)**2 - 4447042639101.56)*cos(11*phi)

@torch.jit.script
def Yl17_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.69491420208903e-15*(1.0 - cos(theta)**2)**6*(5.27721655896904e+16*cos(theta)**5 - 1.59915653302092e+16*cos(theta)**3 + 773785419203672.0*cos(theta))*cos(12*phi)

@torch.jit.script
def Yl17_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.09936771746562e-16*(1.0 - cos(theta)**2)**6.5*(2.63860827948452e+17*cos(theta)**4 - 4.79746959906277e+16*cos(theta)**2 + 773785419203672.0)*cos(13*phi)

@torch.jit.script
def Yl17_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.37542041547274e-17*(1.0 - cos(theta)**2)**7*(1.05544331179381e+18*cos(theta)**3 - 9.59493919812553e+16*cos(theta))*cos(14*phi)

@torch.jit.script
def Yl17_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.50688621401289e-18*(1.0 - cos(theta)**2)**7.5*(3.16632993538143e+18*cos(theta)**2 - 9.59493919812553e+16)*cos(15*phi)

@torch.jit.script
def Yl17_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.07209532485536*(1.0 - cos(theta)**2)**8*cos(16*phi)*cos(theta)

@torch.jit.script
def Yl17_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.869857171920628*(1.0 - cos(theta)**2)**8.5*cos(17*phi)

@torch.jit.script
def Yl18_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.881855768678329*(1.0 - cos(theta)**2)**9*sin(18*phi)

@torch.jit.script
def Yl18_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.29113461206997*(1.0 - cos(theta)**2)**8.5*sin(17*phi)*cos(theta)

@torch.jit.script
def Yl18_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.99730147939357e-19*(1.0 - cos(theta)**2)**8*(1.1082154773835e+20*cos(theta)**2 - 3.16632993538143e+18)*sin(16*phi)

@torch.jit.script
def Yl18_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.01717561545333e-18*(1.0 - cos(theta)**2)**7.5*(3.69405159127833e+19*cos(theta)**3 - 3.16632993538143e+18*cos(theta))*sin(15*phi)

@torch.jit.script
def Yl18_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.31755833840811e-17*(1.0 - cos(theta)**2)**7*(9.23512897819582e+18*cos(theta)**4 - 1.58316496769071e+18*cos(theta)**2 + 2.39873479953138e+16)*sin(14*phi)

@torch.jit.script
def Yl18_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.93150518387396e-16*(1.0 - cos(theta)**2)**6.5*(1.84702579563916e+18*cos(theta)**5 - 5.27721655896904e+17*cos(theta)**3 + 2.39873479953138e+16*cos(theta))*sin(13*phi)

@torch.jit.script
def Yl18_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.9980400343329e-15*(1.0 - cos(theta)**2)**6*(3.07837632606527e+17*cos(theta)**6 - 1.31930413974226e+17*cos(theta)**4 + 1.19936739976569e+16*cos(theta)**2 - 128964236533945.0)*sin(12*phi)

@torch.jit.script
def Yl18_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.79371043838662e-14*(1.0 - cos(theta)**2)**5.5*(4.39768046580754e+16*cos(theta)**7 - 2.63860827948452e+16*cos(theta)**5 + 3.9978913325523e+15*cos(theta)**3 - 128964236533945.0*cos(theta))*sin(11*phi)

@torch.jit.script
def Yl18_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.82471682796557e-13*(1.0 - cos(theta)**2)**5*(5.49710058225942e+15*cos(theta)**8 - 4.39768046580754e+15*cos(theta)**6 + 999472833138076.0*cos(theta)**4 - 64482118266972.7*cos(theta)**2 + 555880329887.695)*sin(10*phi)

@torch.jit.script
def Yl18_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.40088036704182e-11*(1.0 - cos(theta)**2)**4.5*(610788953584380.0*cos(theta)**9 - 628240066543934.0*cos(theta)**7 + 199894566627615.0*cos(theta)**5 - 21494039422324.2*cos(theta)**3 + 555880329887.695*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl18_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.30188133218476e-10*(1.0 - cos(theta)**2)**4*(61078895358438.0*cos(theta)**10 - 78530008317991.7*cos(theta)**8 + 33315761104602.5*cos(theta)**6 - 5373509855581.05*cos(theta)**4 + 277940164943.848*cos(theta)**2 - 2058816036.62109)*sin(8*phi)

@torch.jit.script
def Yl18_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.8928345622358e-9*(1.0 - cos(theta)**2)**3.5*(5552626850767.09*cos(theta)**11 - 8725556479776.86*cos(theta)**9 + 4759394443514.65*cos(theta)**7 - 1074701971116.21*cos(theta)**5 + 92646721647.9492*cos(theta)**3 - 2058816036.62109*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl18_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.74258724725256e-8*(1.0 - cos(theta)**2)**3*(462718904230.591*cos(theta)**12 - 872555647977.686*cos(theta)**10 + 594924305439.331*cos(theta)**8 - 179116995186.035*cos(theta)**6 + 23161680411.9873*cos(theta)**4 - 1029408018.31055*cos(theta)**2 + 6862720.12207031)*sin(6*phi)

@torch.jit.script
def Yl18_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.19097836376173e-6*(1.0 - cos(theta)**2)**2.5*(35593761863.8916*cos(theta)**13 - 79323240725.2441*cos(theta)**11 + 66102700604.3701*cos(theta)**9 - 25588142169.4336*cos(theta)**7 + 4632336082.39746*cos(theta)**5 - 343136006.103516*cos(theta)**3 + 6862720.12207031*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl18_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.13713426594923e-5*(1.0 - cos(theta)**2)**2*(2542411561.70654*cos(theta)**14 - 6610270060.43701*cos(theta)**12 + 6610270060.43701*cos(theta)**10 - 3198517771.1792*cos(theta)**8 + 772056013.73291*cos(theta)**6 - 85784001.5258789*cos(theta)**4 + 3431360.06103516*cos(theta)**2 - 21312.7954101563)*sin(4*phi)

@torch.jit.script
def Yl18_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000388229719023305*(1.0 - cos(theta)**2)**1.5*(169494104.11377*cos(theta)**15 - 508482312.341309*cos(theta)**13 + 600933641.85791*cos(theta)**11 - 355390863.464355*cos(theta)**9 + 110293716.247559*cos(theta)**7 - 17156800.3051758*cos(theta)**5 + 1143786.68701172*cos(theta)**3 - 21312.7954101563*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl18_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00711636829782292*(1.0 - cos(theta)**2)*(10593381.5071106*cos(theta)**16 - 36320165.1672363*cos(theta)**14 + 50077803.4881592*cos(theta)**12 - 35539086.3464355*cos(theta)**10 + 13786714.5309448*cos(theta)**8 - 2859466.7175293*cos(theta)**6 + 285946.67175293*cos(theta)**4 - 10656.3977050781*cos(theta)**2 + 63.4309387207031)*sin(2*phi)

@torch.jit.script
def Yl18_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.131219347792496*(1.0 - cos(theta)**2)**0.5*(623140.088653564*cos(theta)**17 - 2421344.34448242*cos(theta)**15 + 3852138.7298584*cos(theta)**13 - 3230826.03149414*cos(theta)**11 + 1531857.17010498*cos(theta)**9 - 408495.245361328*cos(theta)**7 + 57189.3343505859*cos(theta)**5 - 3552.13256835938*cos(theta)**3 + 63.4309387207031*cos(theta))*sin(phi)

@torch.jit.script
def Yl18_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 59403.1009679377*cos(theta)**18 - 259676.412802699*cos(theta)**16 + 472138.932368544*cos(theta)**14 - 461985.406941263*cos(theta)**12 + 262853.766018305*cos(theta)**10 - 87617.9220061016*cos(theta)**8 + 16355.345441139*cos(theta)**6 - 1523.7899479322*cos(theta)**4 + 54.4210695690072*cos(theta)**2 - 0.318251868824604

@torch.jit.script
def Yl18_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.131219347792496*(1.0 - cos(theta)**2)**0.5*(623140.088653564*cos(theta)**17 - 2421344.34448242*cos(theta)**15 + 3852138.7298584*cos(theta)**13 - 3230826.03149414*cos(theta)**11 + 1531857.17010498*cos(theta)**9 - 408495.245361328*cos(theta)**7 + 57189.3343505859*cos(theta)**5 - 3552.13256835938*cos(theta)**3 + 63.4309387207031*cos(theta))*cos(phi)

@torch.jit.script
def Yl18_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00711636829782292*(1.0 - cos(theta)**2)*(10593381.5071106*cos(theta)**16 - 36320165.1672363*cos(theta)**14 + 50077803.4881592*cos(theta)**12 - 35539086.3464355*cos(theta)**10 + 13786714.5309448*cos(theta)**8 - 2859466.7175293*cos(theta)**6 + 285946.67175293*cos(theta)**4 - 10656.3977050781*cos(theta)**2 + 63.4309387207031)*cos(2*phi)

@torch.jit.script
def Yl18_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000388229719023305*(1.0 - cos(theta)**2)**1.5*(169494104.11377*cos(theta)**15 - 508482312.341309*cos(theta)**13 + 600933641.85791*cos(theta)**11 - 355390863.464355*cos(theta)**9 + 110293716.247559*cos(theta)**7 - 17156800.3051758*cos(theta)**5 + 1143786.68701172*cos(theta)**3 - 21312.7954101563*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl18_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.13713426594923e-5*(1.0 - cos(theta)**2)**2*(2542411561.70654*cos(theta)**14 - 6610270060.43701*cos(theta)**12 + 6610270060.43701*cos(theta)**10 - 3198517771.1792*cos(theta)**8 + 772056013.73291*cos(theta)**6 - 85784001.5258789*cos(theta)**4 + 3431360.06103516*cos(theta)**2 - 21312.7954101563)*cos(4*phi)

@torch.jit.script
def Yl18_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.19097836376173e-6*(1.0 - cos(theta)**2)**2.5*(35593761863.8916*cos(theta)**13 - 79323240725.2441*cos(theta)**11 + 66102700604.3701*cos(theta)**9 - 25588142169.4336*cos(theta)**7 + 4632336082.39746*cos(theta)**5 - 343136006.103516*cos(theta)**3 + 6862720.12207031*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl18_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.74258724725256e-8*(1.0 - cos(theta)**2)**3*(462718904230.591*cos(theta)**12 - 872555647977.686*cos(theta)**10 + 594924305439.331*cos(theta)**8 - 179116995186.035*cos(theta)**6 + 23161680411.9873*cos(theta)**4 - 1029408018.31055*cos(theta)**2 + 6862720.12207031)*cos(6*phi)

@torch.jit.script
def Yl18_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.8928345622358e-9*(1.0 - cos(theta)**2)**3.5*(5552626850767.09*cos(theta)**11 - 8725556479776.86*cos(theta)**9 + 4759394443514.65*cos(theta)**7 - 1074701971116.21*cos(theta)**5 + 92646721647.9492*cos(theta)**3 - 2058816036.62109*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl18_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.30188133218476e-10*(1.0 - cos(theta)**2)**4*(61078895358438.0*cos(theta)**10 - 78530008317991.7*cos(theta)**8 + 33315761104602.5*cos(theta)**6 - 5373509855581.05*cos(theta)**4 + 277940164943.848*cos(theta)**2 - 2058816036.62109)*cos(8*phi)

@torch.jit.script
def Yl18_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.40088036704182e-11*(1.0 - cos(theta)**2)**4.5*(610788953584380.0*cos(theta)**9 - 628240066543934.0*cos(theta)**7 + 199894566627615.0*cos(theta)**5 - 21494039422324.2*cos(theta)**3 + 555880329887.695*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl18_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.82471682796557e-13*(1.0 - cos(theta)**2)**5*(5.49710058225942e+15*cos(theta)**8 - 4.39768046580754e+15*cos(theta)**6 + 999472833138076.0*cos(theta)**4 - 64482118266972.7*cos(theta)**2 + 555880329887.695)*cos(10*phi)

@torch.jit.script
def Yl18_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.79371043838662e-14*(1.0 - cos(theta)**2)**5.5*(4.39768046580754e+16*cos(theta)**7 - 2.63860827948452e+16*cos(theta)**5 + 3.9978913325523e+15*cos(theta)**3 - 128964236533945.0*cos(theta))*cos(11*phi)

@torch.jit.script
def Yl18_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.9980400343329e-15*(1.0 - cos(theta)**2)**6*(3.07837632606527e+17*cos(theta)**6 - 1.31930413974226e+17*cos(theta)**4 + 1.19936739976569e+16*cos(theta)**2 - 128964236533945.0)*cos(12*phi)

@torch.jit.script
def Yl18_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.93150518387396e-16*(1.0 - cos(theta)**2)**6.5*(1.84702579563916e+18*cos(theta)**5 - 5.27721655896904e+17*cos(theta)**3 + 2.39873479953138e+16*cos(theta))*cos(13*phi)

@torch.jit.script
def Yl18_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.31755833840811e-17*(1.0 - cos(theta)**2)**7*(9.23512897819582e+18*cos(theta)**4 - 1.58316496769071e+18*cos(theta)**2 + 2.39873479953138e+16)*cos(14*phi)

@torch.jit.script
def Yl18_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.01717561545333e-18*(1.0 - cos(theta)**2)**7.5*(3.69405159127833e+19*cos(theta)**3 - 3.16632993538143e+18*cos(theta))*cos(15*phi)

@torch.jit.script
def Yl18_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.99730147939357e-19*(1.0 - cos(theta)**2)**8*(1.1082154773835e+20*cos(theta)**2 - 3.16632993538143e+18)*cos(16*phi)

@torch.jit.script
def Yl18_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.29113461206997*(1.0 - cos(theta)**2)**8.5*cos(17*phi)*cos(theta)

@torch.jit.script
def Yl18_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.881855768678329*(1.0 - cos(theta)**2)**9*cos(18*phi)

@torch.jit.script
def Yl19_m_minus_19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.893383784349949*(1.0 - cos(theta)**2)**9.5*sin(19*phi)

@torch.jit.script
def Yl19_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.50718751027224*(1.0 - cos(theta)**2)**9*sin(18*phi)*cos(theta)

@torch.jit.script
def Yl19_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.77683273022057e-21*(1.0 - cos(theta)**2)**8.5*(4.10039726631895e+21*cos(theta)**2 - 1.1082154773835e+20)*sin(17*phi)

@torch.jit.script
def Yl19_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.00346067734132e-20*(1.0 - cos(theta)**2)**8*(1.36679908877298e+21*cos(theta)**3 - 1.1082154773835e+20*cos(theta))*sin(16*phi)

@torch.jit.script
def Yl19_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.1033904683705e-19*(1.0 - cos(theta)**2)**7.5*(3.41699772193245e+20*cos(theta)**4 - 5.54107738691749e+19*cos(theta)**2 + 7.91582483845356e+17)*sin(15*phi)

@torch.jit.script
def Yl19_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.26168804529891e-18*(1.0 - cos(theta)**2)**7*(6.83399544386491e+19*cos(theta)**5 - 1.84702579563916e+19*cos(theta)**3 + 7.91582483845356e+17*cos(theta))*sin(14*phi)

@torch.jit.script
def Yl19_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.30323502710715e-16*(1.0 - cos(theta)**2)**6.5*(1.13899924064415e+19*cos(theta)**6 - 4.61756448909791e+18*cos(theta)**4 + 3.95791241922678e+17*cos(theta)**2 - 3.9978913325523e+15)*sin(13*phi)

@torch.jit.script
def Yl19_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.9505035863512e-15*(1.0 - cos(theta)**2)**6*(1.62714177234879e+18*cos(theta)**7 - 9.23512897819582e+17*cos(theta)**5 + 1.31930413974226e+17*cos(theta)**3 - 3.9978913325523e+15*cos(theta))*sin(12*phi)

@torch.jit.script
def Yl19_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.07165611944352e-14*(1.0 - cos(theta)**2)**5.5*(2.03392721543598e+17*cos(theta)**8 - 1.53918816303264e+17*cos(theta)**6 + 3.29826034935565e+16*cos(theta)**4 - 1.99894566627615e+15*cos(theta)**2 + 16120529566743.2)*sin(11*phi)

@torch.jit.script
def Yl19_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.047246036554e-13*(1.0 - cos(theta)**2)**5*(2.25991912826221e+16*cos(theta)**9 - 2.19884023290377e+16*cos(theta)**7 + 6.5965206987113e+15*cos(theta)**5 - 666315222092051.0*cos(theta)**3 + 16120529566743.2*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl19_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.59515028403688e-12*(1.0 - cos(theta)**2)**4.5*(2.25991912826221e+15*cos(theta)**10 - 2.74855029112971e+15*cos(theta)**8 + 1.09942011645188e+15*cos(theta)**6 - 166578805523013.0*cos(theta)**4 + 8060264783371.58*cos(theta)**2 - 55588032988.7695)*sin(9*phi)

@torch.jit.script
def Yl19_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.50844275293414e-10*(1.0 - cos(theta)**2)**4*(205447193478382.0*cos(theta)**11 - 305394476792190.0*cos(theta)**9 + 157060016635983.0*cos(theta)**7 - 33315761104602.5*cos(theta)**5 + 2686754927790.53*cos(theta)**3 - 55588032988.7695*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl19_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.71519695528145e-9*(1.0 - cos(theta)**2)**3.5*(17120599456531.9*cos(theta)**12 - 30539447679219.0*cos(theta)**10 + 19632502079497.9*cos(theta)**8 - 5552626850767.09*cos(theta)**6 + 671688731947.632*cos(theta)**4 - 27794016494.3848*cos(theta)**2 + 171568003.051758)*sin(7*phi)

@torch.jit.script
def Yl19_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.99182886627511e-8*(1.0 - cos(theta)**2)**3*(1316969188963.99*cos(theta)**13 - 2776313425383.54*cos(theta)**11 + 2181389119944.21*cos(theta)**9 - 793232407252.441*cos(theta)**7 + 134337746389.526*cos(theta)**5 - 9264672164.79492*cos(theta)**3 + 171568003.051758*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl19_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.33885667550482e-7*(1.0 - cos(theta)**2)**2.5*(94069227783.1421*cos(theta)**14 - 231359452115.295*cos(theta)**12 + 218138911994.421*cos(theta)**10 - 99154050906.5552*cos(theta)**8 + 22389624398.2544*cos(theta)**6 - 2316168041.19873*cos(theta)**4 + 85784001.5258789*cos(theta)**2 - 490194.294433594)*sin(5*phi)

@torch.jit.script
def Yl19_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.77192347018779e-5*(1.0 - cos(theta)**2)**2*(6271281852.20947*cos(theta)**15 - 17796880931.9458*cos(theta)**13 + 19830810181.311*cos(theta)**11 - 11017116767.395*cos(theta)**9 + 3198517771.1792*cos(theta)**7 - 463233608.239746*cos(theta)**5 + 28594667.175293*cos(theta)**3 - 490194.294433594*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl19_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000339913857408971*(1.0 - cos(theta)**2)**1.5*(391955115.763092*cos(theta)**16 - 1271205780.85327*cos(theta)**14 + 1652567515.10925*cos(theta)**12 - 1101711676.7395*cos(theta)**10 + 399814721.3974*cos(theta)**8 - 77205601.373291*cos(theta)**6 + 7148666.79382324*cos(theta)**4 - 245097.147216797*cos(theta)**2 + 1332.04971313477)*sin(3*phi)

@torch.jit.script
def Yl19_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00657362114755131*(1.0 - cos(theta)**2)*(23056183.2801819*cos(theta)**17 - 84747052.0568848*cos(theta)**15 + 127120578.085327*cos(theta)**13 - 100155606.976318*cos(theta)**11 + 44423857.9330444*cos(theta)**9 - 11029371.6247559*cos(theta)**7 + 1429733.35876465*cos(theta)**5 - 81699.0490722656*cos(theta)**3 + 1332.04971313477*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl19_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.127805802320551*(1.0 - cos(theta)**2)**0.5*(1280899.07112122*cos(theta)**18 - 5296690.7535553*cos(theta)**16 + 9080041.29180908*cos(theta)**14 - 8346300.58135986*cos(theta)**12 + 4442385.79330444*cos(theta)**10 - 1378671.45309448*cos(theta)**8 + 238288.893127441*cos(theta)**6 - 20424.7622680664*cos(theta)**4 + 666.024856567383*cos(theta)**2 - 3.52394104003906)*sin(phi)

@torch.jit.script
def Yl19_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 118765.056929642*cos(theta)**19 - 548887.154999156*cos(theta)**17 + 1066409.32971265*cos(theta)**15 - 1131040.19818008*cos(theta)**13 + 711460.769822953*cos(theta)**11 - 269864.429932844*cos(theta)**9 + 59969.8733184099*cos(theta)**7 - 7196.38479820918*cos(theta)**5 + 391.10786946789*cos(theta)**3 - 6.20806142012525*cos(theta)

@torch.jit.script
def Yl19_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.127805802320551*(1.0 - cos(theta)**2)**0.5*(1280899.07112122*cos(theta)**18 - 5296690.7535553*cos(theta)**16 + 9080041.29180908*cos(theta)**14 - 8346300.58135986*cos(theta)**12 + 4442385.79330444*cos(theta)**10 - 1378671.45309448*cos(theta)**8 + 238288.893127441*cos(theta)**6 - 20424.7622680664*cos(theta)**4 + 666.024856567383*cos(theta)**2 - 3.52394104003906)*cos(phi)

@torch.jit.script
def Yl19_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00657362114755131*(1.0 - cos(theta)**2)*(23056183.2801819*cos(theta)**17 - 84747052.0568848*cos(theta)**15 + 127120578.085327*cos(theta)**13 - 100155606.976318*cos(theta)**11 + 44423857.9330444*cos(theta)**9 - 11029371.6247559*cos(theta)**7 + 1429733.35876465*cos(theta)**5 - 81699.0490722656*cos(theta)**3 + 1332.04971313477*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl19_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000339913857408971*(1.0 - cos(theta)**2)**1.5*(391955115.763092*cos(theta)**16 - 1271205780.85327*cos(theta)**14 + 1652567515.10925*cos(theta)**12 - 1101711676.7395*cos(theta)**10 + 399814721.3974*cos(theta)**8 - 77205601.373291*cos(theta)**6 + 7148666.79382324*cos(theta)**4 - 245097.147216797*cos(theta)**2 + 1332.04971313477)*cos(3*phi)

@torch.jit.script
def Yl19_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.77192347018779e-5*(1.0 - cos(theta)**2)**2*(6271281852.20947*cos(theta)**15 - 17796880931.9458*cos(theta)**13 + 19830810181.311*cos(theta)**11 - 11017116767.395*cos(theta)**9 + 3198517771.1792*cos(theta)**7 - 463233608.239746*cos(theta)**5 + 28594667.175293*cos(theta)**3 - 490194.294433594*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl19_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.33885667550482e-7*(1.0 - cos(theta)**2)**2.5*(94069227783.1421*cos(theta)**14 - 231359452115.295*cos(theta)**12 + 218138911994.421*cos(theta)**10 - 99154050906.5552*cos(theta)**8 + 22389624398.2544*cos(theta)**6 - 2316168041.19873*cos(theta)**4 + 85784001.5258789*cos(theta)**2 - 490194.294433594)*cos(5*phi)

@torch.jit.script
def Yl19_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.99182886627511e-8*(1.0 - cos(theta)**2)**3*(1316969188963.99*cos(theta)**13 - 2776313425383.54*cos(theta)**11 + 2181389119944.21*cos(theta)**9 - 793232407252.441*cos(theta)**7 + 134337746389.526*cos(theta)**5 - 9264672164.79492*cos(theta)**3 + 171568003.051758*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl19_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.71519695528145e-9*(1.0 - cos(theta)**2)**3.5*(17120599456531.9*cos(theta)**12 - 30539447679219.0*cos(theta)**10 + 19632502079497.9*cos(theta)**8 - 5552626850767.09*cos(theta)**6 + 671688731947.632*cos(theta)**4 - 27794016494.3848*cos(theta)**2 + 171568003.051758)*cos(7*phi)

@torch.jit.script
def Yl19_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.50844275293414e-10*(1.0 - cos(theta)**2)**4*(205447193478382.0*cos(theta)**11 - 305394476792190.0*cos(theta)**9 + 157060016635983.0*cos(theta)**7 - 33315761104602.5*cos(theta)**5 + 2686754927790.53*cos(theta)**3 - 55588032988.7695*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl19_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.59515028403688e-12*(1.0 - cos(theta)**2)**4.5*(2.25991912826221e+15*cos(theta)**10 - 2.74855029112971e+15*cos(theta)**8 + 1.09942011645188e+15*cos(theta)**6 - 166578805523013.0*cos(theta)**4 + 8060264783371.58*cos(theta)**2 - 55588032988.7695)*cos(9*phi)

@torch.jit.script
def Yl19_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.047246036554e-13*(1.0 - cos(theta)**2)**5*(2.25991912826221e+16*cos(theta)**9 - 2.19884023290377e+16*cos(theta)**7 + 6.5965206987113e+15*cos(theta)**5 - 666315222092051.0*cos(theta)**3 + 16120529566743.2*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl19_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.07165611944352e-14*(1.0 - cos(theta)**2)**5.5*(2.03392721543598e+17*cos(theta)**8 - 1.53918816303264e+17*cos(theta)**6 + 3.29826034935565e+16*cos(theta)**4 - 1.99894566627615e+15*cos(theta)**2 + 16120529566743.2)*cos(11*phi)

@torch.jit.script
def Yl19_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.9505035863512e-15*(1.0 - cos(theta)**2)**6*(1.62714177234879e+18*cos(theta)**7 - 9.23512897819582e+17*cos(theta)**5 + 1.31930413974226e+17*cos(theta)**3 - 3.9978913325523e+15*cos(theta))*cos(12*phi)

@torch.jit.script
def Yl19_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.30323502710715e-16*(1.0 - cos(theta)**2)**6.5*(1.13899924064415e+19*cos(theta)**6 - 4.61756448909791e+18*cos(theta)**4 + 3.95791241922678e+17*cos(theta)**2 - 3.9978913325523e+15)*cos(13*phi)

@torch.jit.script
def Yl19_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.26168804529891e-18*(1.0 - cos(theta)**2)**7*(6.83399544386491e+19*cos(theta)**5 - 1.84702579563916e+19*cos(theta)**3 + 7.91582483845356e+17*cos(theta))*cos(14*phi)

@torch.jit.script
def Yl19_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.1033904683705e-19*(1.0 - cos(theta)**2)**7.5*(3.41699772193245e+20*cos(theta)**4 - 5.54107738691749e+19*cos(theta)**2 + 7.91582483845356e+17)*cos(15*phi)

@torch.jit.script
def Yl19_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.00346067734132e-20*(1.0 - cos(theta)**2)**8*(1.36679908877298e+21*cos(theta)**3 - 1.1082154773835e+20*cos(theta))*cos(16*phi)

@torch.jit.script
def Yl19_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.77683273022057e-21*(1.0 - cos(theta)**2)**8.5*(4.10039726631895e+21*cos(theta)**2 - 1.1082154773835e+20)*cos(17*phi)

@torch.jit.script
def Yl19_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.50718751027224*(1.0 - cos(theta)**2)**9*cos(18*phi)*cos(theta)

@torch.jit.script
def Yl19_m19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.893383784349949*(1.0 - cos(theta)**2)**9.5*cos(19*phi)

@torch.jit.script
def Yl20_m_minus_20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.904482145093491*(1.0 - cos(theta)**2)**10*sin(20*phi)

@torch.jit.script
def Yl20_m_minus_19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.72044736290064*(1.0 - cos(theta)**2)**9.5*sin(19*phi)*cos(theta)

@torch.jit.script
def Yl20_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.57963503371958e-22*(1.0 - cos(theta)**2)**9*(1.59915493386439e+23*cos(theta)**2 - 4.10039726631895e+21)*sin(18*phi)

@torch.jit.script
def Yl20_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.68658868646741e-21*(1.0 - cos(theta)**2)**8.5*(5.33051644621463e+22*cos(theta)**3 - 4.10039726631895e+21*cos(theta))*sin(17*phi)

@torch.jit.script
def Yl20_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.05182369321377e-20*(1.0 - cos(theta)**2)**8*(1.33262911155366e+22*cos(theta)**4 - 2.05019863315947e+21*cos(theta)**2 + 2.77053869345875e+19)*sin(16*phi)

@torch.jit.script
def Yl20_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.7528103535224e-19*(1.0 - cos(theta)**2)**7.5*(2.66525822310731e+21*cos(theta)**5 - 6.83399544386491e+20*cos(theta)**3 + 2.77053869345875e+19*cos(theta))*sin(15*phi)

@torch.jit.script
def Yl20_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.9892011943704e-18*(1.0 - cos(theta)**2)**7*(4.44209703851219e+20*cos(theta)**6 - 1.70849886096623e+20*cos(theta)**4 + 1.38526934672937e+19*cos(theta)**2 - 1.31930413974226e+17)*sin(14*phi)

@torch.jit.script
def Yl20_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.15423986229134e-17*(1.0 - cos(theta)**2)**6.5*(6.34585291216027e+19*cos(theta)**7 - 3.41699772193245e+19*cos(theta)**5 + 4.61756448909791e+18*cos(theta)**3 - 1.31930413974226e+17*cos(theta))*sin(13*phi)

@torch.jit.script
def Yl20_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.99945619851927e-16*(1.0 - cos(theta)**2)**6*(7.93231614020034e+18*cos(theta)**8 - 5.69499620322076e+18*cos(theta)**6 + 1.15439112227448e+18*cos(theta)**4 - 6.5965206987113e+16*cos(theta)**2 + 499736416569038.0)*sin(12*phi)

@torch.jit.script
def Yl20_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.6969639886762e-14*(1.0 - cos(theta)**2)**5.5*(8.8136846002226e+17*cos(theta)**9 - 8.13570886174394e+17*cos(theta)**7 + 2.30878224454896e+17*cos(theta)**5 - 2.19884023290377e+16*cos(theta)**3 + 499736416569038.0*cos(theta))*sin(11*phi)

@torch.jit.script
def Yl20_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.98781341694522e-13*(1.0 - cos(theta)**2)**5*(8.8136846002226e+16*cos(theta)**10 - 1.01696360771799e+17*cos(theta)**8 + 3.84797040758159e+16*cos(theta)**6 - 5.49710058225942e+15*cos(theta)**4 + 249868208284519.0*cos(theta)**2 - 1612052956674.32)*sin(10*phi)

@torch.jit.script
def Yl20_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.42763260987486e-12*(1.0 - cos(theta)**2)**4.5*(8.01244054565691e+15*cos(theta)**11 - 1.1299595641311e+16*cos(theta)**9 + 5.49710058225942e+15*cos(theta)**7 - 1.09942011645188e+15*cos(theta)**5 + 83289402761506.3*cos(theta)**3 - 1612052956674.32*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl20_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.01251173426417e-10*(1.0 - cos(theta)**2)**4*(667703378804743.0*cos(theta)**12 - 1.1299595641311e+15*cos(theta)**10 + 687137572782427.0*cos(theta)**8 - 183236686075314.0*cos(theta)**6 + 20822350690376.6*cos(theta)**4 - 806026478337.158*cos(theta)**2 + 4632336082.39746)*sin(8*phi)

@torch.jit.script
def Yl20_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.9317492704185e-9*(1.0 - cos(theta)**2)**3.5*(51361798369595.6*cos(theta)**13 - 102723596739191.0*cos(theta)**11 + 76348619198047.5*cos(theta)**9 - 26176669439330.6*cos(theta)**7 + 4164470138075.32*cos(theta)**5 - 268675492779.053*cos(theta)**3 + 4632336082.39746*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl20_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.75574983477626e-8*(1.0 - cos(theta)**2)**3*(3668699883542.54*cos(theta)**14 - 8560299728265.93*cos(theta)**12 + 7634861919804.75*cos(theta)**10 - 3272083679916.32*cos(theta)**8 + 694078356345.886*cos(theta)**6 - 67168873194.7632*cos(theta)**4 + 2316168041.19873*cos(theta)**2 - 12254857.3608398)*sin(6*phi)

@torch.jit.script
def Yl20_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.417011635662e-7*(1.0 - cos(theta)**2)**2.5*(244579992236.169*cos(theta)**15 - 658484594481.995*cos(theta)**13 + 694078356345.886*cos(theta)**11 - 363564853324.036*cos(theta)**9 + 99154050906.5552*cos(theta)**7 - 13433774638.9526*cos(theta)**5 + 772056013.73291*cos(theta)**3 - 12254857.3608398*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl20_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.4834023271324e-5*(1.0 - cos(theta)**2)**2*(15286249514.7606*cos(theta)**16 - 47034613891.571*cos(theta)**14 + 57839863028.8239*cos(theta)**12 - 36356485332.4036*cos(theta)**10 + 12394256363.3194*cos(theta)**8 - 2238962439.82544*cos(theta)**6 + 193014003.433228*cos(theta)**4 - 6127428.68041992*cos(theta)**2 + 30637.1434020996)*sin(4*phi)

@torch.jit.script
def Yl20_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000299632582569029*(1.0 - cos(theta)**2)**1.5*(899191147.927094*cos(theta)**17 - 3135640926.10474*cos(theta)**15 + 4449220232.98645*cos(theta)**13 - 3305135030.21851*cos(theta)**11 + 1377139595.92438*cos(theta)**9 - 319851777.11792*cos(theta)**7 + 38602800.6866455*cos(theta)**5 - 2042476.22680664*cos(theta)**3 + 30637.1434020996*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl20_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00609662114603756*(1.0 - cos(theta)**2)*(49955063.7737274*cos(theta)**18 - 195977557.881546*cos(theta)**16 + 317801445.213318*cos(theta)**14 - 275427919.184875*cos(theta)**12 + 137713959.592438*cos(theta)**10 - 39981472.13974*cos(theta)**8 + 6433800.11444092*cos(theta)**6 - 510619.05670166*cos(theta)**4 + 15318.5717010498*cos(theta)**2 - 74.0027618408203)*sin(2*phi)

@torch.jit.script
def Yl20_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.12464571379913*(1.0 - cos(theta)**2)**0.5*(2629213.88282776*cos(theta)**19 - 11528091.6400909*cos(theta)**17 + 21186763.0142212*cos(theta)**15 - 21186763.0142212*cos(theta)**13 + 12519450.8720398*cos(theta)**11 - 4442385.79330444*cos(theta)**9 + 919114.302062988*cos(theta)**7 - 102123.811340332*cos(theta)**5 + 5106.1905670166*cos(theta)**3 - 74.0027618408203*cos(theta))*sin(phi)

@torch.jit.script
def Yl20_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 237455.874096927*cos(theta)**20 - 1156836.30970298*cos(theta)**18 + 2391837.23492643*cos(theta)**16 - 2733528.26848735*cos(theta)**14 + 1884477.82145719*cos(theta)**12 - 802422.814297898*cos(theta)**10 + 207523.141628767*cos(theta)**8 - 30744.1691301877*cos(theta)**6 + 2305.81268476408*cos(theta)**4 - 66.8351502830167*cos(theta)**2 + 0.318262620395318

@torch.jit.script
def Yl20_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.12464571379913*(1.0 - cos(theta)**2)**0.5*(2629213.88282776*cos(theta)**19 - 11528091.6400909*cos(theta)**17 + 21186763.0142212*cos(theta)**15 - 21186763.0142212*cos(theta)**13 + 12519450.8720398*cos(theta)**11 - 4442385.79330444*cos(theta)**9 + 919114.302062988*cos(theta)**7 - 102123.811340332*cos(theta)**5 + 5106.1905670166*cos(theta)**3 - 74.0027618408203*cos(theta))*cos(phi)

@torch.jit.script
def Yl20_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00609662114603756*(1.0 - cos(theta)**2)*(49955063.7737274*cos(theta)**18 - 195977557.881546*cos(theta)**16 + 317801445.213318*cos(theta)**14 - 275427919.184875*cos(theta)**12 + 137713959.592438*cos(theta)**10 - 39981472.13974*cos(theta)**8 + 6433800.11444092*cos(theta)**6 - 510619.05670166*cos(theta)**4 + 15318.5717010498*cos(theta)**2 - 74.0027618408203)*cos(2*phi)

@torch.jit.script
def Yl20_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000299632582569029*(1.0 - cos(theta)**2)**1.5*(899191147.927094*cos(theta)**17 - 3135640926.10474*cos(theta)**15 + 4449220232.98645*cos(theta)**13 - 3305135030.21851*cos(theta)**11 + 1377139595.92438*cos(theta)**9 - 319851777.11792*cos(theta)**7 + 38602800.6866455*cos(theta)**5 - 2042476.22680664*cos(theta)**3 + 30637.1434020996*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl20_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.4834023271324e-5*(1.0 - cos(theta)**2)**2*(15286249514.7606*cos(theta)**16 - 47034613891.571*cos(theta)**14 + 57839863028.8239*cos(theta)**12 - 36356485332.4036*cos(theta)**10 + 12394256363.3194*cos(theta)**8 - 2238962439.82544*cos(theta)**6 + 193014003.433228*cos(theta)**4 - 6127428.68041992*cos(theta)**2 + 30637.1434020996)*cos(4*phi)

@torch.jit.script
def Yl20_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.417011635662e-7*(1.0 - cos(theta)**2)**2.5*(244579992236.169*cos(theta)**15 - 658484594481.995*cos(theta)**13 + 694078356345.886*cos(theta)**11 - 363564853324.036*cos(theta)**9 + 99154050906.5552*cos(theta)**7 - 13433774638.9526*cos(theta)**5 + 772056013.73291*cos(theta)**3 - 12254857.3608398*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl20_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.75574983477626e-8*(1.0 - cos(theta)**2)**3*(3668699883542.54*cos(theta)**14 - 8560299728265.93*cos(theta)**12 + 7634861919804.75*cos(theta)**10 - 3272083679916.32*cos(theta)**8 + 694078356345.886*cos(theta)**6 - 67168873194.7632*cos(theta)**4 + 2316168041.19873*cos(theta)**2 - 12254857.3608398)*cos(6*phi)

@torch.jit.script
def Yl20_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.9317492704185e-9*(1.0 - cos(theta)**2)**3.5*(51361798369595.6*cos(theta)**13 - 102723596739191.0*cos(theta)**11 + 76348619198047.5*cos(theta)**9 - 26176669439330.6*cos(theta)**7 + 4164470138075.32*cos(theta)**5 - 268675492779.053*cos(theta)**3 + 4632336082.39746*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl20_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.01251173426417e-10*(1.0 - cos(theta)**2)**4*(667703378804743.0*cos(theta)**12 - 1.1299595641311e+15*cos(theta)**10 + 687137572782427.0*cos(theta)**8 - 183236686075314.0*cos(theta)**6 + 20822350690376.6*cos(theta)**4 - 806026478337.158*cos(theta)**2 + 4632336082.39746)*cos(8*phi)

@torch.jit.script
def Yl20_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.42763260987486e-12*(1.0 - cos(theta)**2)**4.5*(8.01244054565691e+15*cos(theta)**11 - 1.1299595641311e+16*cos(theta)**9 + 5.49710058225942e+15*cos(theta)**7 - 1.09942011645188e+15*cos(theta)**5 + 83289402761506.3*cos(theta)**3 - 1612052956674.32*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl20_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.98781341694522e-13*(1.0 - cos(theta)**2)**5*(8.8136846002226e+16*cos(theta)**10 - 1.01696360771799e+17*cos(theta)**8 + 3.84797040758159e+16*cos(theta)**6 - 5.49710058225942e+15*cos(theta)**4 + 249868208284519.0*cos(theta)**2 - 1612052956674.32)*cos(10*phi)

@torch.jit.script
def Yl20_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.6969639886762e-14*(1.0 - cos(theta)**2)**5.5*(8.8136846002226e+17*cos(theta)**9 - 8.13570886174394e+17*cos(theta)**7 + 2.30878224454896e+17*cos(theta)**5 - 2.19884023290377e+16*cos(theta)**3 + 499736416569038.0*cos(theta))*cos(11*phi)

@torch.jit.script
def Yl20_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.99945619851927e-16*(1.0 - cos(theta)**2)**6*(7.93231614020034e+18*cos(theta)**8 - 5.69499620322076e+18*cos(theta)**6 + 1.15439112227448e+18*cos(theta)**4 - 6.5965206987113e+16*cos(theta)**2 + 499736416569038.0)*cos(12*phi)

@torch.jit.script
def Yl20_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.15423986229134e-17*(1.0 - cos(theta)**2)**6.5*(6.34585291216027e+19*cos(theta)**7 - 3.41699772193245e+19*cos(theta)**5 + 4.61756448909791e+18*cos(theta)**3 - 1.31930413974226e+17*cos(theta))*cos(13*phi)

@torch.jit.script
def Yl20_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.9892011943704e-18*(1.0 - cos(theta)**2)**7*(4.44209703851219e+20*cos(theta)**6 - 1.70849886096623e+20*cos(theta)**4 + 1.38526934672937e+19*cos(theta)**2 - 1.31930413974226e+17)*cos(14*phi)

@torch.jit.script
def Yl20_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.7528103535224e-19*(1.0 - cos(theta)**2)**7.5*(2.66525822310731e+21*cos(theta)**5 - 6.83399544386491e+20*cos(theta)**3 + 2.77053869345875e+19*cos(theta))*cos(15*phi)

@torch.jit.script
def Yl20_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.05182369321377e-20*(1.0 - cos(theta)**2)**8*(1.33262911155366e+22*cos(theta)**4 - 2.05019863315947e+21*cos(theta)**2 + 2.77053869345875e+19)*cos(16*phi)

@torch.jit.script
def Yl20_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.68658868646741e-21*(1.0 - cos(theta)**2)**8.5*(5.33051644621463e+22*cos(theta)**3 - 4.10039726631895e+21*cos(theta))*cos(17*phi)

@torch.jit.script
def Yl20_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.57963503371958e-22*(1.0 - cos(theta)**2)**9*(1.59915493386439e+23*cos(theta)**2 - 4.10039726631895e+21)*cos(18*phi)

@torch.jit.script
def Yl20_m19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.72044736290064*(1.0 - cos(theta)**2)**9.5*cos(19*phi)*cos(theta)

@torch.jit.script
def Yl20_m20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.904482145093491*(1.0 - cos(theta)**2)**10*cos(20*phi)

@torch.jit.script
def Yl21_m_minus_21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.915186448400331*(1.0 - cos(theta)**2)**10.5*sin(21*phi)

@torch.jit.script
def Yl21_m_minus_20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.93108606277937*(1.0 - cos(theta)**2)**10*sin(20*phi)*cos(theta)

@torch.jit.script
def Yl21_m_minus_19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.09578128625229e-24*(1.0 - cos(theta)**2)**9.5*(6.55653522884399e+24*cos(theta)**2 - 1.59915493386439e+23)*sin(19*phi)

@torch.jit.script
def Yl21_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.48670360217581e-23*(1.0 - cos(theta)**2)**9*(2.185511742948e+24*cos(theta)**3 - 1.59915493386439e+23*cos(theta))*sin(18*phi)

@torch.jit.script
def Yl21_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.60389100299896e-22*(1.0 - cos(theta)**2)**8.5*(5.46377935737e+23*cos(theta)**4 - 7.99577466932194e+22*cos(theta)**2 + 1.02509931657974e+21)*sin(17*phi)

@torch.jit.script
def Yl21_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.72443067867375e-21*(1.0 - cos(theta)**2)**8*(1.092755871474e+23*cos(theta)**5 - 2.66525822310731e+22*cos(theta)**3 + 1.02509931657974e+21*cos(theta))*sin(16*phi)

@torch.jit.script
def Yl21_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.15091424992218e-19*(1.0 - cos(theta)**2)**7.5*(1.82125978579e+22*cos(theta)**6 - 6.66314555776829e+21*cos(theta)**4 + 5.12549658289868e+20*cos(theta)**2 - 4.61756448909791e+18)*sin(15*phi)

@torch.jit.script
def Yl21_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.82701973139271e-18*(1.0 - cos(theta)**2)**7*(2.60179969398571e+21*cos(theta)**7 - 1.33262911155366e+21*cos(theta)**5 + 1.70849886096623e+20*cos(theta)**3 - 4.61756448909791e+18*cos(theta))*sin(14*phi)

@torch.jit.script
def Yl21_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.05718875389061e-17*(1.0 - cos(theta)**2)**6.5*(3.25224961748214e+20*cos(theta)**8 - 2.2210485192561e+20*cos(theta)**6 + 4.27124715241557e+19*cos(theta)**4 - 2.30878224454896e+18*cos(theta)**2 + 1.64913017467783e+16)*sin(13*phi)

@torch.jit.script
def Yl21_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.34789616721945e-16*(1.0 - cos(theta)**2)**6*(3.61361068609127e+19*cos(theta)**9 - 3.17292645608014e+19*cos(theta)**7 + 8.54249430483114e+18*cos(theta)**5 - 7.69594081516319e+17*cos(theta)**3 + 1.64913017467783e+16*cos(theta))*sin(12*phi)

@torch.jit.script
def Yl21_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.71493583461516e-15*(1.0 - cos(theta)**2)**5.5*(3.61361068609127e+18*cos(theta)**10 - 3.96615807010017e+18*cos(theta)**8 + 1.42374905080519e+18*cos(theta)**6 - 1.9239852037908e+17*cos(theta)**4 + 8.24565087338913e+15*cos(theta)**2 - 49973641656903.8)*sin(11*phi)

@torch.jit.script
def Yl21_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.82268352577409e-13*(1.0 - cos(theta)**2)**5*(3.28510062371933e+17*cos(theta)**11 - 4.4068423001113e+17*cos(theta)**9 + 2.03392721543598e+17*cos(theta)**7 - 3.84797040758159e+16*cos(theta)**5 + 2.74855029112971e+15*cos(theta)**3 - 49973641656903.8*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl21_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.51546467407613e-12*(1.0 - cos(theta)**2)**4.5*(2.73758385309944e+16*cos(theta)**12 - 4.4068423001113e+16*cos(theta)**10 + 2.54240901929498e+16*cos(theta)**8 - 6.41328401263599e+15*cos(theta)**6 + 687137572782427.0*cos(theta)**4 - 24986820828451.9*cos(theta)**2 + 134337746389.526)*sin(9*phi)

@torch.jit.script
def Yl21_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.94248646460625e-11*(1.0 - cos(theta)**2)**4*(2.10583373315342e+15*cos(theta)**13 - 4.00622027282846e+15*cos(theta)**11 + 2.82489891032776e+15*cos(theta)**9 - 916183430376570.0*cos(theta)**7 + 137427514556485.0*cos(theta)**5 - 8328940276150.63*cos(theta)**3 + 134337746389.526*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl21_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.39887226130065e-9*(1.0 - cos(theta)**2)**3.5*(150416695225244.0*cos(theta)**14 - 333851689402371.0*cos(theta)**12 + 282489891032776.0*cos(theta)**10 - 114522928797071.0*cos(theta)**8 + 22904585759414.2*cos(theta)**6 - 2082235069037.66*cos(theta)**4 + 67168873194.7632*cos(theta)**2 - 330881148.742676)*sin(7*phi)

@torch.jit.script
def Yl21_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.86683503788286e-8*(1.0 - cos(theta)**2)**3*(10027779681682.9*cos(theta)**15 - 25680899184797.8*cos(theta)**13 + 25680899184797.8*cos(theta)**11 - 12724769866341.2*cos(theta)**9 + 3272083679916.32*cos(theta)**7 - 416447013807.532*cos(theta)**5 + 22389624398.2544*cos(theta)**3 - 330881148.742676*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl21_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.95860473103812e-7*(1.0 - cos(theta)**2)**2.5*(626736230105.184*cos(theta)**16 - 1834349941771.27*cos(theta)**14 + 2140074932066.48*cos(theta)**12 - 1272476986634.12*cos(theta)**10 + 409010459989.54*cos(theta)**8 - 69407835634.5886*cos(theta)**6 + 5597406099.5636*cos(theta)**4 - 165440574.371338*cos(theta)**2 + 765928.58505249)*sin(5*phi)

@torch.jit.script
def Yl21_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.25272490558029e-5*(1.0 - cos(theta)**2)**2*(36866837065.0108*cos(theta)**17 - 122289996118.085*cos(theta)**15 + 164621148620.499*cos(theta)**13 - 115679726057.648*cos(theta)**11 + 45445606665.5045*cos(theta)**9 - 9915405090.65552*cos(theta)**7 + 1119481219.91272*cos(theta)**5 - 55146858.1237793*cos(theta)**3 + 765928.58505249*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl21_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00026574308270913*(1.0 - cos(theta)**2)**1.5*(2048157614.72282*cos(theta)**18 - 7643124757.38029*cos(theta)**16 + 11758653472.8928*cos(theta)**14 - 9639977171.47064*cos(theta)**12 + 4544560666.55045*cos(theta)**10 - 1239425636.33194*cos(theta)**8 + 186580203.318787*cos(theta)**6 - 13786714.5309448*cos(theta)**4 + 382964.292526245*cos(theta)**2 - 1702.06352233887)*sin(3*phi)

@torch.jit.script
def Yl21_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00567471937804281*(1.0 - cos(theta)**2)*(107797769.195938*cos(theta)**19 - 449595573.963547*cos(theta)**17 + 783910231.526184*cos(theta)**15 - 741536705.497742*cos(theta)**13 + 413141878.777313*cos(theta)**11 - 137713959.592438*cos(theta)**9 + 26654314.7598267*cos(theta)**7 - 2757342.90618896*cos(theta)**5 + 127654.764175415*cos(theta)**3 - 1702.06352233887*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl21_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.121709171425106*(1.0 - cos(theta)**2)**0.5*(5389888.45979691*cos(theta)**20 - 24977531.8868637*cos(theta)**18 + 48994389.4703865*cos(theta)**16 - 52966907.535553*cos(theta)**14 + 34428489.8981094*cos(theta)**12 - 13771395.9592438*cos(theta)**10 + 3331789.34497833*cos(theta)**8 - 459557.151031494*cos(theta)**6 + 31913.6910438538*cos(theta)**4 - 851.031761169434*cos(theta)**2 + 3.70013809204102)*sin(phi)

@torch.jit.script
def Yl21_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 474777.116937231*cos(theta)**21 - 2431785.23309314*cos(theta)**19 + 5331221.47255034*cos(theta)**17 - 6531947.02943104*cos(theta)**15 + 4898960.27207328*cos(theta)**13 - 2315872.12861646*cos(theta)**11 + 684800.898246803*cos(theta)**9 - 121442.523827019*cos(theta)**7 + 11806.912038738*cos(theta)**5 - 524.751646166133*cos(theta)**3 + 6.84458668912347*cos(theta)

@torch.jit.script
def Yl21_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.121709171425106*(1.0 - cos(theta)**2)**0.5*(5389888.45979691*cos(theta)**20 - 24977531.8868637*cos(theta)**18 + 48994389.4703865*cos(theta)**16 - 52966907.535553*cos(theta)**14 + 34428489.8981094*cos(theta)**12 - 13771395.9592438*cos(theta)**10 + 3331789.34497833*cos(theta)**8 - 459557.151031494*cos(theta)**6 + 31913.6910438538*cos(theta)**4 - 851.031761169434*cos(theta)**2 + 3.70013809204102)*cos(phi)

@torch.jit.script
def Yl21_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00567471937804281*(1.0 - cos(theta)**2)*(107797769.195938*cos(theta)**19 - 449595573.963547*cos(theta)**17 + 783910231.526184*cos(theta)**15 - 741536705.497742*cos(theta)**13 + 413141878.777313*cos(theta)**11 - 137713959.592438*cos(theta)**9 + 26654314.7598267*cos(theta)**7 - 2757342.90618896*cos(theta)**5 + 127654.764175415*cos(theta)**3 - 1702.06352233887*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl21_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00026574308270913*(1.0 - cos(theta)**2)**1.5*(2048157614.72282*cos(theta)**18 - 7643124757.38029*cos(theta)**16 + 11758653472.8928*cos(theta)**14 - 9639977171.47064*cos(theta)**12 + 4544560666.55045*cos(theta)**10 - 1239425636.33194*cos(theta)**8 + 186580203.318787*cos(theta)**6 - 13786714.5309448*cos(theta)**4 + 382964.292526245*cos(theta)**2 - 1702.06352233887)*cos(3*phi)

@torch.jit.script
def Yl21_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.25272490558029e-5*(1.0 - cos(theta)**2)**2*(36866837065.0108*cos(theta)**17 - 122289996118.085*cos(theta)**15 + 164621148620.499*cos(theta)**13 - 115679726057.648*cos(theta)**11 + 45445606665.5045*cos(theta)**9 - 9915405090.65552*cos(theta)**7 + 1119481219.91272*cos(theta)**5 - 55146858.1237793*cos(theta)**3 + 765928.58505249*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl21_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.95860473103812e-7*(1.0 - cos(theta)**2)**2.5*(626736230105.184*cos(theta)**16 - 1834349941771.27*cos(theta)**14 + 2140074932066.48*cos(theta)**12 - 1272476986634.12*cos(theta)**10 + 409010459989.54*cos(theta)**8 - 69407835634.5886*cos(theta)**6 + 5597406099.5636*cos(theta)**4 - 165440574.371338*cos(theta)**2 + 765928.58505249)*cos(5*phi)

@torch.jit.script
def Yl21_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.86683503788286e-8*(1.0 - cos(theta)**2)**3*(10027779681682.9*cos(theta)**15 - 25680899184797.8*cos(theta)**13 + 25680899184797.8*cos(theta)**11 - 12724769866341.2*cos(theta)**9 + 3272083679916.32*cos(theta)**7 - 416447013807.532*cos(theta)**5 + 22389624398.2544*cos(theta)**3 - 330881148.742676*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl21_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.39887226130065e-9*(1.0 - cos(theta)**2)**3.5*(150416695225244.0*cos(theta)**14 - 333851689402371.0*cos(theta)**12 + 282489891032776.0*cos(theta)**10 - 114522928797071.0*cos(theta)**8 + 22904585759414.2*cos(theta)**6 - 2082235069037.66*cos(theta)**4 + 67168873194.7632*cos(theta)**2 - 330881148.742676)*cos(7*phi)

@torch.jit.script
def Yl21_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.94248646460625e-11*(1.0 - cos(theta)**2)**4*(2.10583373315342e+15*cos(theta)**13 - 4.00622027282846e+15*cos(theta)**11 + 2.82489891032776e+15*cos(theta)**9 - 916183430376570.0*cos(theta)**7 + 137427514556485.0*cos(theta)**5 - 8328940276150.63*cos(theta)**3 + 134337746389.526*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl21_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.51546467407613e-12*(1.0 - cos(theta)**2)**4.5*(2.73758385309944e+16*cos(theta)**12 - 4.4068423001113e+16*cos(theta)**10 + 2.54240901929498e+16*cos(theta)**8 - 6.41328401263599e+15*cos(theta)**6 + 687137572782427.0*cos(theta)**4 - 24986820828451.9*cos(theta)**2 + 134337746389.526)*cos(9*phi)

@torch.jit.script
def Yl21_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.82268352577409e-13*(1.0 - cos(theta)**2)**5*(3.28510062371933e+17*cos(theta)**11 - 4.4068423001113e+17*cos(theta)**9 + 2.03392721543598e+17*cos(theta)**7 - 3.84797040758159e+16*cos(theta)**5 + 2.74855029112971e+15*cos(theta)**3 - 49973641656903.8*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl21_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.71493583461516e-15*(1.0 - cos(theta)**2)**5.5*(3.61361068609127e+18*cos(theta)**10 - 3.96615807010017e+18*cos(theta)**8 + 1.42374905080519e+18*cos(theta)**6 - 1.9239852037908e+17*cos(theta)**4 + 8.24565087338913e+15*cos(theta)**2 - 49973641656903.8)*cos(11*phi)

@torch.jit.script
def Yl21_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.34789616721945e-16*(1.0 - cos(theta)**2)**6*(3.61361068609127e+19*cos(theta)**9 - 3.17292645608014e+19*cos(theta)**7 + 8.54249430483114e+18*cos(theta)**5 - 7.69594081516319e+17*cos(theta)**3 + 1.64913017467783e+16*cos(theta))*cos(12*phi)

@torch.jit.script
def Yl21_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.05718875389061e-17*(1.0 - cos(theta)**2)**6.5*(3.25224961748214e+20*cos(theta)**8 - 2.2210485192561e+20*cos(theta)**6 + 4.27124715241557e+19*cos(theta)**4 - 2.30878224454896e+18*cos(theta)**2 + 1.64913017467783e+16)*cos(13*phi)

@torch.jit.script
def Yl21_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.82701973139271e-18*(1.0 - cos(theta)**2)**7*(2.60179969398571e+21*cos(theta)**7 - 1.33262911155366e+21*cos(theta)**5 + 1.70849886096623e+20*cos(theta)**3 - 4.61756448909791e+18*cos(theta))*cos(14*phi)

@torch.jit.script
def Yl21_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.15091424992218e-19*(1.0 - cos(theta)**2)**7.5*(1.82125978579e+22*cos(theta)**6 - 6.66314555776829e+21*cos(theta)**4 + 5.12549658289868e+20*cos(theta)**2 - 4.61756448909791e+18)*cos(15*phi)

@torch.jit.script
def Yl21_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.72443067867375e-21*(1.0 - cos(theta)**2)**8*(1.092755871474e+23*cos(theta)**5 - 2.66525822310731e+22*cos(theta)**3 + 1.02509931657974e+21*cos(theta))*cos(16*phi)

@torch.jit.script
def Yl21_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.60389100299896e-22*(1.0 - cos(theta)**2)**8.5*(5.46377935737e+23*cos(theta)**4 - 7.99577466932194e+22*cos(theta)**2 + 1.02509931657974e+21)*cos(17*phi)

@torch.jit.script
def Yl21_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.48670360217581e-23*(1.0 - cos(theta)**2)**9*(2.185511742948e+24*cos(theta)**3 - 1.59915493386439e+23*cos(theta))*cos(18*phi)

@torch.jit.script
def Yl21_m19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.09578128625229e-24*(1.0 - cos(theta)**2)**9.5*(6.55653522884399e+24*cos(theta)**2 - 1.59915493386439e+23)*cos(19*phi)

@torch.jit.script
def Yl21_m20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.93108606277937*(1.0 - cos(theta)**2)**10*cos(20*phi)*cos(theta)

@torch.jit.script
def Yl21_m21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.915186448400331*(1.0 - cos(theta)**2)**10.5*cos(21*phi)

@torch.jit.script
def Yl22_m_minus_22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.925527866459589*(1.0 - cos(theta)**2)**11*sin(22*phi)

@torch.jit.script
def Yl22_m_minus_21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.13925733212923*(1.0 - cos(theta)**2)**10.5*sin(21*phi)*cos(theta)

@torch.jit.script
def Yl22_m_minus_20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.00969966670912e-25*(1.0 - cos(theta)**2)**10*(2.81931014840292e+26*cos(theta)**2 - 6.55653522884399e+24)*sin(20*phi)

@torch.jit.script
def Yl22_m_minus_19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.13338506490961e-24*(1.0 - cos(theta)**2)**9.5*(9.39770049467639e+25*cos(theta)**3 - 6.55653522884399e+24*cos(theta))*sin(19*phi)

@torch.jit.script
def Yl22_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.45144107589342e-23*(1.0 - cos(theta)**2)**9*(2.3494251236691e+25*cos(theta)**4 - 3.278267614422e+24*cos(theta)**2 + 3.99788733466097e+22)*sin(18*phi)

@torch.jit.script
def Yl22_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.05264765451388e-22*(1.0 - cos(theta)**2)**8.5*(4.6988502473382e+24*cos(theta)**5 - 1.092755871474e+24*cos(theta)**3 + 3.99788733466097e+22*cos(theta))*sin(17*phi)

@torch.jit.script
def Yl22_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.13994713346901e-21*(1.0 - cos(theta)**2)**8*(7.83141707889699e+23*cos(theta)**6 - 2.731889678685e+23*cos(theta)**4 + 1.99894366733049e+22*cos(theta)**2 - 1.70849886096623e+20)*sin(16*phi)

@torch.jit.script
def Yl22_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.12109879641152e-20*(1.0 - cos(theta)**2)**7.5*(1.11877386841386e+23*cos(theta)**7 - 5.46377935737e+22*cos(theta)**5 + 6.66314555776829e+21*cos(theta)**3 - 1.70849886096623e+20*cos(theta))*sin(15*phi)

@torch.jit.script
def Yl22_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.81067151427848e-19*(1.0 - cos(theta)**2)**7*(1.39846733551732e+22*cos(theta)**8 - 9.10629892894999e+21*cos(theta)**6 + 1.66578638944207e+21*cos(theta)**4 - 8.54249430483114e+19*cos(theta)**2 + 5.77195561137239e+17)*sin(14*phi)

@torch.jit.script
def Yl22_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.58592087257013e-17*(1.0 - cos(theta)**2)**6.5*(1.55385259501924e+21*cos(theta)**9 - 1.30089984699286e+21*cos(theta)**7 + 3.33157277888414e+20*cos(theta)**5 - 2.84749810161038e+19*cos(theta)**3 + 5.77195561137239e+17*cos(theta))*sin(13*phi)

@torch.jit.script
def Yl22_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.9669862738455e-16*(1.0 - cos(theta)**2)**6*(1.55385259501924e+20*cos(theta)**10 - 1.62612480874107e+20*cos(theta)**8 + 5.55262129814024e+19*cos(theta)**6 - 7.11874525402595e+18*cos(theta)**4 + 2.8859778056862e+17*cos(theta)**2 - 1.64913017467783e+15)*sin(12*phi)

@torch.jit.script
def Yl22_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.73787837392547e-15*(1.0 - cos(theta)**2)**5.5*(1.41259326819931e+19*cos(theta)**11 - 1.80680534304563e+19*cos(theta)**9 + 7.93231614020034e+18*cos(theta)**7 - 1.42374905080519e+18*cos(theta)**5 + 9.61992601895398e+16*cos(theta)**3 - 1.64913017467783e+15*cos(theta))*sin(11*phi)

@torch.jit.script
def Yl22_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.14182337954032e-13*(1.0 - cos(theta)**2)**5*(1.17716105683276e+18*cos(theta)**12 - 1.80680534304563e+18*cos(theta)**10 + 9.91539517525043e+17*cos(theta)**8 - 2.37291508467532e+17*cos(theta)**6 + 2.4049815047385e+16*cos(theta)**4 - 824565087338913.0*cos(theta)**2 + 4164470138075.32)*sin(10*phi)

@torch.jit.script
def Yl22_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.32887187734102e-12*(1.0 - cos(theta)**2)**4.5*(9.0550850525597e+16*cos(theta)**13 - 1.64255031185967e+17*cos(theta)**11 + 1.10171057502783e+17*cos(theta)**9 - 3.38987869239331e+16*cos(theta)**7 + 4.80996300947699e+15*cos(theta)**5 - 274855029112971.0*cos(theta)**3 + 4164470138075.32*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl22_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.85166115051776e-11*(1.0 - cos(theta)**2)**4*(6.4679178946855e+15*cos(theta)**14 - 1.36879192654972e+16*cos(theta)**12 + 1.10171057502783e+16*cos(theta)**10 - 4.23734836549164e+15*cos(theta)**8 + 801660501579499.0*cos(theta)**6 - 68713757278242.7*cos(theta)**4 + 2082235069037.66*cos(theta)**2 - 9595553313.5376)*sin(8*phi)

@torch.jit.script
def Yl22_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.02919274986513e-9*(1.0 - cos(theta)**2)**3.5*(431194526312367.0*cos(theta)**15 - 1.05291686657671e+15*cos(theta)**13 + 1.00155506820711e+15*cos(theta)**11 - 470816485054626.0*cos(theta)**9 + 114522928797071.0*cos(theta)**7 - 13742751455648.5*cos(theta)**5 + 694078356345.886*cos(theta)**3 - 9595553313.5376*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl22_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.21694903053267e-8*(1.0 - cos(theta)**2)**3*(26949657894522.9*cos(theta)**16 - 75208347612622.1*cos(theta)**14 + 83462922350592.8*cos(theta)**12 - 47081648505462.6*cos(theta)**10 + 14315366099633.9*cos(theta)**8 - 2290458575941.42*cos(theta)**6 + 173519589086.472*cos(theta)**4 - 4797776656.7688*cos(theta)**2 + 20680071.7964172)*sin(6*phi)

@torch.jit.script
def Yl22_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.83681174938034e-7*(1.0 - cos(theta)**2)**2.5*(1585273993795.47*cos(theta)**17 - 5013889840841.47*cos(theta)**15 + 6420224796199.45*cos(theta)**13 - 4280149864132.96*cos(theta)**11 + 1590596233292.66*cos(theta)**9 - 327208367991.632*cos(theta)**7 + 34703917817.2943*cos(theta)**5 - 1599258885.5896*cos(theta)**3 + 20680071.7964172*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl22_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.06629486910923e-5*(1.0 - cos(theta)**2)**2*(88070777433.0814*cos(theta)**18 - 313368115052.592*cos(theta)**16 + 458587485442.818*cos(theta)**14 - 356679155344.414*cos(theta)**12 + 159059623329.266*cos(theta)**10 - 40901045998.954*cos(theta)**8 + 5783986302.88239*cos(theta)**6 - 399814721.3974*cos(theta)**4 + 10340035.8982086*cos(theta)**2 - 42551.5880584717)*sin(4*phi)

@torch.jit.script
def Yl22_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000236995878752564*(1.0 - cos(theta)**2)**1.5*(4635304075.42534*cos(theta)**19 - 18433418532.5054*cos(theta)**17 + 30572499029.5212*cos(theta)**15 - 27436858103.4164*cos(theta)**13 + 14459965757.206*cos(theta)**11 - 4544560666.55045*cos(theta)**9 + 826283757.554626*cos(theta)**7 - 79962944.27948*cos(theta)**5 + 3446678.63273621*cos(theta)**3 - 42551.5880584717*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl22_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00529938895278031*(1.0 - cos(theta)**2)*(231765203.771267*cos(theta)**20 - 1024078807.36141*cos(theta)**18 + 1910781189.34507*cos(theta)**16 - 1959775578.81546*cos(theta)**14 + 1204997146.43383*cos(theta)**12 - 454456066.655045*cos(theta)**10 + 103285469.694328*cos(theta)**8 - 13327157.3799133*cos(theta)**6 + 861669.658184052*cos(theta)**4 - 21275.7940292358*cos(theta)**2 + 85.1031761169434)*sin(2*phi)

@torch.jit.script
def Yl22_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.118970986923352*(1.0 - cos(theta)**2)**0.5*(11036438.2748222*cos(theta)**21 - 53898884.5979691*cos(theta)**19 + 112398893.490887*cos(theta)**17 - 130651705.254364*cos(theta)**15 + 92692088.1872177*cos(theta)**13 - 41314187.8777313*cos(theta)**11 + 11476163.2993698*cos(theta)**9 - 1903879.6257019*cos(theta)**7 + 172333.93163681*cos(theta)**5 - 7091.93134307861*cos(theta)**3 + 85.1031761169434*cos(theta))*sin(phi)

@torch.jit.script
def Yl22_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 949308.966084274*cos(theta)**22 - 5099776.07361552*cos(theta)**20 + 11816554.316914*cos(theta)**18 - 15452417.1836568*cos(theta)**16 + 12528986.9056677*cos(theta)**14 - 6515073.19094719*cos(theta)**12 + 2171691.06364906*cos(theta)**10 - 450350.681401879*cos(theta)**8 + 54352.6684450544*cos(theta)**6 - 3355.10299043546*cos(theta)**4 + 80.5224717704509*cos(theta)**2 - 0.318270639408897

@torch.jit.script
def Yl22_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.118970986923352*(1.0 - cos(theta)**2)**0.5*(11036438.2748222*cos(theta)**21 - 53898884.5979691*cos(theta)**19 + 112398893.490887*cos(theta)**17 - 130651705.254364*cos(theta)**15 + 92692088.1872177*cos(theta)**13 - 41314187.8777313*cos(theta)**11 + 11476163.2993698*cos(theta)**9 - 1903879.6257019*cos(theta)**7 + 172333.93163681*cos(theta)**5 - 7091.93134307861*cos(theta)**3 + 85.1031761169434*cos(theta))*cos(phi)

@torch.jit.script
def Yl22_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00529938895278031*(1.0 - cos(theta)**2)*(231765203.771267*cos(theta)**20 - 1024078807.36141*cos(theta)**18 + 1910781189.34507*cos(theta)**16 - 1959775578.81546*cos(theta)**14 + 1204997146.43383*cos(theta)**12 - 454456066.655045*cos(theta)**10 + 103285469.694328*cos(theta)**8 - 13327157.3799133*cos(theta)**6 + 861669.658184052*cos(theta)**4 - 21275.7940292358*cos(theta)**2 + 85.1031761169434)*cos(2*phi)

@torch.jit.script
def Yl22_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000236995878752564*(1.0 - cos(theta)**2)**1.5*(4635304075.42534*cos(theta)**19 - 18433418532.5054*cos(theta)**17 + 30572499029.5212*cos(theta)**15 - 27436858103.4164*cos(theta)**13 + 14459965757.206*cos(theta)**11 - 4544560666.55045*cos(theta)**9 + 826283757.554626*cos(theta)**7 - 79962944.27948*cos(theta)**5 + 3446678.63273621*cos(theta)**3 - 42551.5880584717*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl22_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.06629486910923e-5*(1.0 - cos(theta)**2)**2*(88070777433.0814*cos(theta)**18 - 313368115052.592*cos(theta)**16 + 458587485442.818*cos(theta)**14 - 356679155344.414*cos(theta)**12 + 159059623329.266*cos(theta)**10 - 40901045998.954*cos(theta)**8 + 5783986302.88239*cos(theta)**6 - 399814721.3974*cos(theta)**4 + 10340035.8982086*cos(theta)**2 - 42551.5880584717)*cos(4*phi)

@torch.jit.script
def Yl22_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.83681174938034e-7*(1.0 - cos(theta)**2)**2.5*(1585273993795.47*cos(theta)**17 - 5013889840841.47*cos(theta)**15 + 6420224796199.45*cos(theta)**13 - 4280149864132.96*cos(theta)**11 + 1590596233292.66*cos(theta)**9 - 327208367991.632*cos(theta)**7 + 34703917817.2943*cos(theta)**5 - 1599258885.5896*cos(theta)**3 + 20680071.7964172*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl22_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.21694903053267e-8*(1.0 - cos(theta)**2)**3*(26949657894522.9*cos(theta)**16 - 75208347612622.1*cos(theta)**14 + 83462922350592.8*cos(theta)**12 - 47081648505462.6*cos(theta)**10 + 14315366099633.9*cos(theta)**8 - 2290458575941.42*cos(theta)**6 + 173519589086.472*cos(theta)**4 - 4797776656.7688*cos(theta)**2 + 20680071.7964172)*cos(6*phi)

@torch.jit.script
def Yl22_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.02919274986513e-9*(1.0 - cos(theta)**2)**3.5*(431194526312367.0*cos(theta)**15 - 1.05291686657671e+15*cos(theta)**13 + 1.00155506820711e+15*cos(theta)**11 - 470816485054626.0*cos(theta)**9 + 114522928797071.0*cos(theta)**7 - 13742751455648.5*cos(theta)**5 + 694078356345.886*cos(theta)**3 - 9595553313.5376*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl22_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.85166115051776e-11*(1.0 - cos(theta)**2)**4*(6.4679178946855e+15*cos(theta)**14 - 1.36879192654972e+16*cos(theta)**12 + 1.10171057502783e+16*cos(theta)**10 - 4.23734836549164e+15*cos(theta)**8 + 801660501579499.0*cos(theta)**6 - 68713757278242.7*cos(theta)**4 + 2082235069037.66*cos(theta)**2 - 9595553313.5376)*cos(8*phi)

@torch.jit.script
def Yl22_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.32887187734102e-12*(1.0 - cos(theta)**2)**4.5*(9.0550850525597e+16*cos(theta)**13 - 1.64255031185967e+17*cos(theta)**11 + 1.10171057502783e+17*cos(theta)**9 - 3.38987869239331e+16*cos(theta)**7 + 4.80996300947699e+15*cos(theta)**5 - 274855029112971.0*cos(theta)**3 + 4164470138075.32*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl22_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.14182337954032e-13*(1.0 - cos(theta)**2)**5*(1.17716105683276e+18*cos(theta)**12 - 1.80680534304563e+18*cos(theta)**10 + 9.91539517525043e+17*cos(theta)**8 - 2.37291508467532e+17*cos(theta)**6 + 2.4049815047385e+16*cos(theta)**4 - 824565087338913.0*cos(theta)**2 + 4164470138075.32)*cos(10*phi)

@torch.jit.script
def Yl22_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.73787837392547e-15*(1.0 - cos(theta)**2)**5.5*(1.41259326819931e+19*cos(theta)**11 - 1.80680534304563e+19*cos(theta)**9 + 7.93231614020034e+18*cos(theta)**7 - 1.42374905080519e+18*cos(theta)**5 + 9.61992601895398e+16*cos(theta)**3 - 1.64913017467783e+15*cos(theta))*cos(11*phi)

@torch.jit.script
def Yl22_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.9669862738455e-16*(1.0 - cos(theta)**2)**6*(1.55385259501924e+20*cos(theta)**10 - 1.62612480874107e+20*cos(theta)**8 + 5.55262129814024e+19*cos(theta)**6 - 7.11874525402595e+18*cos(theta)**4 + 2.8859778056862e+17*cos(theta)**2 - 1.64913017467783e+15)*cos(12*phi)

@torch.jit.script
def Yl22_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.58592087257013e-17*(1.0 - cos(theta)**2)**6.5*(1.55385259501924e+21*cos(theta)**9 - 1.30089984699286e+21*cos(theta)**7 + 3.33157277888414e+20*cos(theta)**5 - 2.84749810161038e+19*cos(theta)**3 + 5.77195561137239e+17*cos(theta))*cos(13*phi)

@torch.jit.script
def Yl22_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.81067151427848e-19*(1.0 - cos(theta)**2)**7*(1.39846733551732e+22*cos(theta)**8 - 9.10629892894999e+21*cos(theta)**6 + 1.66578638944207e+21*cos(theta)**4 - 8.54249430483114e+19*cos(theta)**2 + 5.77195561137239e+17)*cos(14*phi)

@torch.jit.script
def Yl22_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.12109879641152e-20*(1.0 - cos(theta)**2)**7.5*(1.11877386841386e+23*cos(theta)**7 - 5.46377935737e+22*cos(theta)**5 + 6.66314555776829e+21*cos(theta)**3 - 1.70849886096623e+20*cos(theta))*cos(15*phi)

@torch.jit.script
def Yl22_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.13994713346901e-21*(1.0 - cos(theta)**2)**8*(7.83141707889699e+23*cos(theta)**6 - 2.731889678685e+23*cos(theta)**4 + 1.99894366733049e+22*cos(theta)**2 - 1.70849886096623e+20)*cos(16*phi)

@torch.jit.script
def Yl22_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.05264765451388e-22*(1.0 - cos(theta)**2)**8.5*(4.6988502473382e+24*cos(theta)**5 - 1.092755871474e+24*cos(theta)**3 + 3.99788733466097e+22*cos(theta))*cos(17*phi)

@torch.jit.script
def Yl22_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.45144107589342e-23*(1.0 - cos(theta)**2)**9*(2.3494251236691e+25*cos(theta)**4 - 3.278267614422e+24*cos(theta)**2 + 3.99788733466097e+22)*cos(18*phi)

@torch.jit.script
def Yl22_m19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.13338506490961e-24*(1.0 - cos(theta)**2)**9.5*(9.39770049467639e+25*cos(theta)**3 - 6.55653522884399e+24*cos(theta))*cos(19*phi)

@torch.jit.script
def Yl22_m20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.00969966670912e-25*(1.0 - cos(theta)**2)**10*(2.81931014840292e+26*cos(theta)**2 - 6.55653522884399e+24)*cos(20*phi)

@torch.jit.script
def Yl22_m21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.13925733212923*(1.0 - cos(theta)**2)**10.5*cos(21*phi)*cos(theta)

@torch.jit.script
def Yl22_m22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.925527866459589*(1.0 - cos(theta)**2)**11*cos(22*phi)

@torch.jit.script
def Yl23_m_minus_23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.935533863919911*(1.0 - cos(theta)**2)**11.5*sin(23*phi)

@torch.jit.script
def Yl23_m_minus_22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.34509937549305*(1.0 - cos(theta)**2)**11*sin(22*phi)*cos(theta)

@torch.jit.script
def Yl23_m_minus_21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.37232572869364e-27*(1.0 - cos(theta)**2)**10.5*(1.26868956678131e+28*cos(theta)**2 - 2.81931014840292e+26)*sin(21*phi)

@torch.jit.script
def Yl23_m_minus_20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.72559475329492e-26*(1.0 - cos(theta)**2)**10*(4.22896522260438e+27*cos(theta)**3 - 2.81931014840292e+26*cos(theta))*sin(20*phi)

@torch.jit.script
def Yl23_m_minus_19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.5745840073783e-25*(1.0 - cos(theta)**2)**9.5*(1.05724130565109e+27*cos(theta)**4 - 1.40965507420146e+26*cos(theta)**2 + 1.639133807211e+24)*sin(19*phi)

@torch.jit.script
def Yl23_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.18006435618226e-24*(1.0 - cos(theta)**2)**9*(2.11448261130219e+26*cos(theta)**5 - 4.6988502473382e+25*cos(theta)**3 + 1.639133807211e+24*cos(theta))*sin(18*phi)

@torch.jit.script
def Yl23_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.12461347795126e-23*(1.0 - cos(theta)**2)**8.5*(3.52413768550365e+25*cos(theta)**6 - 1.17471256183455e+25*cos(theta)**4 + 8.19566903605499e+23*cos(theta)**2 - 6.66314555776829e+21)*sin(17*phi)

@torch.jit.script
def Yl23_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.35950786560836e-21*(1.0 - cos(theta)**2)**8*(5.03448240786235e+24*cos(theta)**7 - 2.3494251236691e+24*cos(theta)**5 + 2.731889678685e+23*cos(theta)**3 - 6.66314555776829e+21*cos(theta))*sin(16*phi)

@torch.jit.script
def Yl23_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.40136967298897e-20*(1.0 - cos(theta)**2)**7.5*(6.29310300982794e+23*cos(theta)**8 - 3.9157085394485e+23*cos(theta)**6 + 6.82972419671249e+22*cos(theta)**4 - 3.33157277888414e+21*cos(theta)**2 + 2.13562357620778e+19)*sin(15*phi)

@torch.jit.script
def Yl23_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.44091105154346e-19*(1.0 - cos(theta)**2)**7*(6.9923366775866e+22*cos(theta)**9 - 5.59386934206928e+22*cos(theta)**7 + 1.3659448393425e+22*cos(theta)**5 - 1.11052425962805e+21*cos(theta)**3 + 2.13562357620778e+19*cos(theta))*sin(14*phi)

@torch.jit.script
def Yl23_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.54226296601593e-18*(1.0 - cos(theta)**2)**6.5*(6.9923366775866e+21*cos(theta)**10 - 6.9923366775866e+21*cos(theta)**8 + 2.2765747322375e+21*cos(theta)**6 - 2.77631064907012e+20*cos(theta)**4 + 1.06781178810389e+19*cos(theta)**2 - 5.77195561137239e+16)*sin(13*phi)

@torch.jit.script
def Yl23_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.6998888671294e-16*(1.0 - cos(theta)**2)**6*(6.35666970689691e+20*cos(theta)**11 - 7.76926297509622e+20*cos(theta)**9 + 3.25224961748214e+20*cos(theta)**7 - 5.55262129814024e+19*cos(theta)**5 + 3.55937262701297e+18*cos(theta)**3 - 5.77195561137239e+16*cos(theta))*sin(12*phi)

@torch.jit.script
def Yl23_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.48373550581555e-15*(1.0 - cos(theta)**2)**5.5*(5.29722475574742e+19*cos(theta)**12 - 7.76926297509622e+19*cos(theta)**10 + 4.06531202185268e+19*cos(theta)**8 - 9.25436883023373e+18*cos(theta)**6 + 8.89843156753243e+17*cos(theta)**4 - 2.88597780568619e+16*cos(theta)**2 + 137427514556485.0)*sin(11*phi)

@torch.jit.script
def Yl23_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.32413447372461e-14*(1.0 - cos(theta)**2)**5*(4.07478827365187e+18*cos(theta)**13 - 7.06296634099657e+18*cos(theta)**11 + 4.51701335761408e+18*cos(theta)**9 - 1.32205269003339e+18*cos(theta)**7 + 1.77968631350649e+17*cos(theta)**5 - 9.61992601895398e+15*cos(theta)**3 + 137427514556485.0*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl23_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.57426303248889e-12*(1.0 - cos(theta)**2)**4.5*(2.91056305260848e+17*cos(theta)**14 - 5.88580528416381e+17*cos(theta)**12 + 4.51701335761408e+17*cos(theta)**10 - 1.65256586254174e+17*cos(theta)**8 + 2.96614385584414e+16*cos(theta)**6 - 2.4049815047385e+15*cos(theta)**4 + 68713757278242.7*cos(theta)**2 - 297462152719.666)*sin(9*phi)

@torch.jit.script
def Yl23_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.4490374973626e-11*(1.0 - cos(theta)**2)**4*(1.94037536840565e+16*cos(theta)**15 - 4.52754252627985e+16*cos(theta)**13 + 4.10637577964917e+16*cos(theta)**11 - 1.83618429171304e+16*cos(theta)**9 + 4.23734836549164e+15*cos(theta)**7 - 480996300947699.0*cos(theta)**5 + 22904585759414.2*cos(theta)**3 - 297462152719.666*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl23_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.68137122555198e-10*(1.0 - cos(theta)**2)**3.5*(1.21273460525353e+15*cos(theta)**16 - 3.23395894734275e+15*cos(theta)**14 + 3.42197981637431e+15*cos(theta)**12 - 1.83618429171304e+15*cos(theta)**10 + 529668545686454.0*cos(theta)**8 - 80166050157949.9*cos(theta)**6 + 5726146439853.56*cos(theta)**4 - 148731076359.833*cos(theta)**2 + 599722082.0961)*sin(7*phi)

@torch.jit.script
def Yl23_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.73469785817059e-8*(1.0 - cos(theta)**2)**3*(71337329720796.0*cos(theta)**17 - 215597263156183.0*cos(theta)**15 + 263229216644177.0*cos(theta)**13 - 166925844701186.0*cos(theta)**11 + 58852060631828.3*cos(theta)**9 - 11452292879707.1*cos(theta)**7 + 1145229287970.71*cos(theta)**5 - 49577025453.2776*cos(theta)**3 + 599722082.0961*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl23_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.96331958851659e-7*(1.0 - cos(theta)**2)**2.5*(3963184984488.66*cos(theta)**18 - 13474828947261.5*cos(theta)**16 + 18802086903155.5*cos(theta)**14 - 13910487058432.1*cos(theta)**12 + 5885206063182.83*cos(theta)**10 - 1431536609963.39*cos(theta)**8 + 190871547995.119*cos(theta)**6 - 12394256363.3194*cos(theta)**4 + 299861041.04805*cos(theta)**2 - 1148892.87757874)*sin(5*phi)

@torch.jit.script
def Yl23_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.1414462474505e-6*(1.0 - cos(theta)**2)**2*(208588683394.14*cos(theta)**19 - 792636996897.733*cos(theta)**17 + 1253472460210.37*cos(theta)**15 - 1070037466033.24*cos(theta)**13 + 535018733016.621*cos(theta)**11 - 159059623329.266*cos(theta)**9 + 27267363999.3027*cos(theta)**7 - 2478851272.66388*cos(theta)**5 + 99953680.34935*cos(theta)**3 - 1148892.87757874*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl23_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000212428014459756*(1.0 - cos(theta)**2)**1.5*(10429434169.707*cos(theta)**20 - 44035388716.5407*cos(theta)**18 + 78342028763.148*cos(theta)**16 - 76431247573.8029*cos(theta)**14 + 44584894418.0517*cos(theta)**12 - 15905962332.9266*cos(theta)**10 + 3408420499.91283*cos(theta)**8 - 413141878.777313*cos(theta)**6 + 24988420.0873375*cos(theta)**4 - 574446.438789368*cos(theta)**2 + 2127.57940292358)*sin(3*phi)

@torch.jit.script
def Yl23_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00496372955394567*(1.0 - cos(theta)**2)*(496639722.367001*cos(theta)**21 - 2317652037.71267*cos(theta)**19 + 4608354633.12635*cos(theta)**17 - 5095416504.9202*cos(theta)**15 + 3429607262.92706*cos(theta)**13 - 1445996575.7206*cos(theta)**11 + 378713388.879204*cos(theta)**9 - 59020268.396759*cos(theta)**7 + 4997684.0174675*cos(theta)**5 - 191482.146263123*cos(theta)**3 + 2127.57940292358*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl23_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.116409776636641*(1.0 - cos(theta)**2)**0.5*(22574532.8348637*cos(theta)**22 - 115882601.885633*cos(theta)**20 + 256019701.840353*cos(theta)**18 - 318463531.557512*cos(theta)**16 + 244971947.351933*cos(theta)**14 - 120499714.643383*cos(theta)**12 + 37871338.8879204*cos(theta)**10 - 7377533.54959488*cos(theta)**8 + 832947.336244583*cos(theta)**6 - 47870.5365657806*cos(theta)**4 + 1063.78970146179*cos(theta)**2 - 3.86832618713379)*sin(phi)

@torch.jit.script
def Yl23_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1898169.24542421*cos(theta)**23 - 10671929.3131628*cos(theta)**21 + 26059362.2763277*cos(theta)**19 - 36228869.5061141*cos(theta)**17 + 31584142.6463559*cos(theta)**15 - 17926135.0154993*cos(theta)**13 + 6658278.72004259*cos(theta)**11 - 1585304.457153*cos(theta)**9 + 230124.840554467*cos(theta)**7 - 18515.7917687503*cos(theta)**5 + 685.770065509269*cos(theta)**3 - 7.48112798737384*cos(theta)

@torch.jit.script
def Yl23_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.116409776636641*(1.0 - cos(theta)**2)**0.5*(22574532.8348637*cos(theta)**22 - 115882601.885633*cos(theta)**20 + 256019701.840353*cos(theta)**18 - 318463531.557512*cos(theta)**16 + 244971947.351933*cos(theta)**14 - 120499714.643383*cos(theta)**12 + 37871338.8879204*cos(theta)**10 - 7377533.54959488*cos(theta)**8 + 832947.336244583*cos(theta)**6 - 47870.5365657806*cos(theta)**4 + 1063.78970146179*cos(theta)**2 - 3.86832618713379)*cos(phi)

@torch.jit.script
def Yl23_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00496372955394567*(1.0 - cos(theta)**2)*(496639722.367001*cos(theta)**21 - 2317652037.71267*cos(theta)**19 + 4608354633.12635*cos(theta)**17 - 5095416504.9202*cos(theta)**15 + 3429607262.92706*cos(theta)**13 - 1445996575.7206*cos(theta)**11 + 378713388.879204*cos(theta)**9 - 59020268.396759*cos(theta)**7 + 4997684.0174675*cos(theta)**5 - 191482.146263123*cos(theta)**3 + 2127.57940292358*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl23_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000212428014459756*(1.0 - cos(theta)**2)**1.5*(10429434169.707*cos(theta)**20 - 44035388716.5407*cos(theta)**18 + 78342028763.148*cos(theta)**16 - 76431247573.8029*cos(theta)**14 + 44584894418.0517*cos(theta)**12 - 15905962332.9266*cos(theta)**10 + 3408420499.91283*cos(theta)**8 - 413141878.777313*cos(theta)**6 + 24988420.0873375*cos(theta)**4 - 574446.438789368*cos(theta)**2 + 2127.57940292358)*cos(3*phi)

@torch.jit.script
def Yl23_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.1414462474505e-6*(1.0 - cos(theta)**2)**2*(208588683394.14*cos(theta)**19 - 792636996897.733*cos(theta)**17 + 1253472460210.37*cos(theta)**15 - 1070037466033.24*cos(theta)**13 + 535018733016.621*cos(theta)**11 - 159059623329.266*cos(theta)**9 + 27267363999.3027*cos(theta)**7 - 2478851272.66388*cos(theta)**5 + 99953680.34935*cos(theta)**3 - 1148892.87757874*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl23_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.96331958851659e-7*(1.0 - cos(theta)**2)**2.5*(3963184984488.66*cos(theta)**18 - 13474828947261.5*cos(theta)**16 + 18802086903155.5*cos(theta)**14 - 13910487058432.1*cos(theta)**12 + 5885206063182.83*cos(theta)**10 - 1431536609963.39*cos(theta)**8 + 190871547995.119*cos(theta)**6 - 12394256363.3194*cos(theta)**4 + 299861041.04805*cos(theta)**2 - 1148892.87757874)*cos(5*phi)

@torch.jit.script
def Yl23_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.73469785817059e-8*(1.0 - cos(theta)**2)**3*(71337329720796.0*cos(theta)**17 - 215597263156183.0*cos(theta)**15 + 263229216644177.0*cos(theta)**13 - 166925844701186.0*cos(theta)**11 + 58852060631828.3*cos(theta)**9 - 11452292879707.1*cos(theta)**7 + 1145229287970.71*cos(theta)**5 - 49577025453.2776*cos(theta)**3 + 599722082.0961*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl23_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.68137122555198e-10*(1.0 - cos(theta)**2)**3.5*(1.21273460525353e+15*cos(theta)**16 - 3.23395894734275e+15*cos(theta)**14 + 3.42197981637431e+15*cos(theta)**12 - 1.83618429171304e+15*cos(theta)**10 + 529668545686454.0*cos(theta)**8 - 80166050157949.9*cos(theta)**6 + 5726146439853.56*cos(theta)**4 - 148731076359.833*cos(theta)**2 + 599722082.0961)*cos(7*phi)

@torch.jit.script
def Yl23_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.4490374973626e-11*(1.0 - cos(theta)**2)**4*(1.94037536840565e+16*cos(theta)**15 - 4.52754252627985e+16*cos(theta)**13 + 4.10637577964917e+16*cos(theta)**11 - 1.83618429171304e+16*cos(theta)**9 + 4.23734836549164e+15*cos(theta)**7 - 480996300947699.0*cos(theta)**5 + 22904585759414.2*cos(theta)**3 - 297462152719.666*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl23_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.57426303248889e-12*(1.0 - cos(theta)**2)**4.5*(2.91056305260848e+17*cos(theta)**14 - 5.88580528416381e+17*cos(theta)**12 + 4.51701335761408e+17*cos(theta)**10 - 1.65256586254174e+17*cos(theta)**8 + 2.96614385584414e+16*cos(theta)**6 - 2.4049815047385e+15*cos(theta)**4 + 68713757278242.7*cos(theta)**2 - 297462152719.666)*cos(9*phi)

@torch.jit.script
def Yl23_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.32413447372461e-14*(1.0 - cos(theta)**2)**5*(4.07478827365187e+18*cos(theta)**13 - 7.06296634099657e+18*cos(theta)**11 + 4.51701335761408e+18*cos(theta)**9 - 1.32205269003339e+18*cos(theta)**7 + 1.77968631350649e+17*cos(theta)**5 - 9.61992601895398e+15*cos(theta)**3 + 137427514556485.0*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl23_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.48373550581555e-15*(1.0 - cos(theta)**2)**5.5*(5.29722475574742e+19*cos(theta)**12 - 7.76926297509622e+19*cos(theta)**10 + 4.06531202185268e+19*cos(theta)**8 - 9.25436883023373e+18*cos(theta)**6 + 8.89843156753243e+17*cos(theta)**4 - 2.88597780568619e+16*cos(theta)**2 + 137427514556485.0)*cos(11*phi)

@torch.jit.script
def Yl23_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.6998888671294e-16*(1.0 - cos(theta)**2)**6*(6.35666970689691e+20*cos(theta)**11 - 7.76926297509622e+20*cos(theta)**9 + 3.25224961748214e+20*cos(theta)**7 - 5.55262129814024e+19*cos(theta)**5 + 3.55937262701297e+18*cos(theta)**3 - 5.77195561137239e+16*cos(theta))*cos(12*phi)

@torch.jit.script
def Yl23_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.54226296601593e-18*(1.0 - cos(theta)**2)**6.5*(6.9923366775866e+21*cos(theta)**10 - 6.9923366775866e+21*cos(theta)**8 + 2.2765747322375e+21*cos(theta)**6 - 2.77631064907012e+20*cos(theta)**4 + 1.06781178810389e+19*cos(theta)**2 - 5.77195561137239e+16)*cos(13*phi)

@torch.jit.script
def Yl23_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.44091105154346e-19*(1.0 - cos(theta)**2)**7*(6.9923366775866e+22*cos(theta)**9 - 5.59386934206928e+22*cos(theta)**7 + 1.3659448393425e+22*cos(theta)**5 - 1.11052425962805e+21*cos(theta)**3 + 2.13562357620778e+19*cos(theta))*cos(14*phi)

@torch.jit.script
def Yl23_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.40136967298897e-20*(1.0 - cos(theta)**2)**7.5*(6.29310300982794e+23*cos(theta)**8 - 3.9157085394485e+23*cos(theta)**6 + 6.82972419671249e+22*cos(theta)**4 - 3.33157277888414e+21*cos(theta)**2 + 2.13562357620778e+19)*cos(15*phi)

@torch.jit.script
def Yl23_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.35950786560836e-21*(1.0 - cos(theta)**2)**8*(5.03448240786235e+24*cos(theta)**7 - 2.3494251236691e+24*cos(theta)**5 + 2.731889678685e+23*cos(theta)**3 - 6.66314555776829e+21*cos(theta))*cos(16*phi)

@torch.jit.script
def Yl23_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.12461347795126e-23*(1.0 - cos(theta)**2)**8.5*(3.52413768550365e+25*cos(theta)**6 - 1.17471256183455e+25*cos(theta)**4 + 8.19566903605499e+23*cos(theta)**2 - 6.66314555776829e+21)*cos(17*phi)

@torch.jit.script
def Yl23_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.18006435618226e-24*(1.0 - cos(theta)**2)**9*(2.11448261130219e+26*cos(theta)**5 - 4.6988502473382e+25*cos(theta)**3 + 1.639133807211e+24*cos(theta))*cos(18*phi)

@torch.jit.script
def Yl23_m19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.5745840073783e-25*(1.0 - cos(theta)**2)**9.5*(1.05724130565109e+27*cos(theta)**4 - 1.40965507420146e+26*cos(theta)**2 + 1.639133807211e+24)*cos(19*phi)

@torch.jit.script
def Yl23_m20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.72559475329492e-26*(1.0 - cos(theta)**2)**10*(4.22896522260438e+27*cos(theta)**3 - 2.81931014840292e+26*cos(theta))*cos(20*phi)

@torch.jit.script
def Yl23_m21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.37232572869364e-27*(1.0 - cos(theta)**2)**10.5*(1.26868956678131e+28*cos(theta)**2 - 2.81931014840292e+26)*cos(21*phi)

@torch.jit.script
def Yl23_m22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.34509937549305*(1.0 - cos(theta)**2)**11*cos(22*phi)*cos(theta)

@torch.jit.script
def Yl23_m23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.935533863919911*(1.0 - cos(theta)**2)**11.5*cos(23*phi)

@torch.jit.script
def Yl24_m_minus_24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.9452287742978*(1.0 - cos(theta)**2)**12*sin(24*phi)

@torch.jit.script
def Yl24_m_minus_23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.54873704743938*(1.0 - cos(theta)**2)**11.5*sin(23*phi)*cos(theta)

@torch.jit.script
def Yl24_m_minus_22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.3240025801011e-29*(1.0 - cos(theta)**2)**11*(5.96284096387217e+29*cos(theta)**2 - 1.26868956678131e+28)*sin(22*phi)

@torch.jit.script
def Yl24_m_minus_21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.25428691320073e-28*(1.0 - cos(theta)**2)**10.5*(1.98761365462406e+29*cos(theta)**3 - 1.26868956678131e+28*cos(theta))*sin(21*phi)

@torch.jit.script
def Yl24_m_minus_20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.39100641322249e-27*(1.0 - cos(theta)**2)**10*(4.96903413656014e+28*cos(theta)**4 - 6.34344783390656e+27*cos(theta)**2 + 7.04827537100729e+25)*sin(20*phi)

@torch.jit.script
def Yl24_m_minus_19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.24458738133901e-25*(1.0 - cos(theta)**2)**9.5*(9.93806827312028e+27*cos(theta)**5 - 2.11448261130219e+27*cos(theta)**3 + 7.04827537100729e+25*cos(theta))*sin(19*phi)

@torch.jit.script
def Yl24_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.99910334761708e-24*(1.0 - cos(theta)**2)**9*(1.65634471218671e+27*cos(theta)**6 - 5.28620652825547e+26*cos(theta)**4 + 3.52413768550365e+25*cos(theta)**2 - 2.731889678685e+23)*sin(18*phi)

@torch.jit.script
def Yl24_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.42774820132609e-23*(1.0 - cos(theta)**2)**8.5*(2.36620673169531e+26*cos(theta)**7 - 1.05724130565109e+26*cos(theta)**5 + 1.17471256183455e+25*cos(theta)**3 - 2.731889678685e+23*cos(theta))*sin(17*phi)

@torch.jit.script
def Yl24_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.2079160239131e-22*(1.0 - cos(theta)**2)**8*(2.95775841461913e+25*cos(theta)**8 - 1.76206884275182e+25*cos(theta)**6 + 2.93678140458637e+24*cos(theta)**4 - 1.3659448393425e+23*cos(theta)**2 + 8.32893194721036e+20)*sin(16*phi)

@torch.jit.script
def Yl24_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.1778692495173e-20*(1.0 - cos(theta)**2)**7.5*(3.2863982384657e+24*cos(theta)**9 - 2.51724120393118e+24*cos(theta)**7 + 5.87356280917274e+23*cos(theta)**5 - 4.553149464475e+22*cos(theta)**3 + 8.32893194721036e+20*cos(theta))*sin(15*phi)

@torch.jit.script
def Yl24_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.32610538861376e-19*(1.0 - cos(theta)**2)**7*(3.2863982384657e+23*cos(theta)**10 - 3.14655150491397e+23*cos(theta)**8 + 9.78927134862124e+22*cos(theta)**6 - 1.13828736611875e+22*cos(theta)**4 + 4.16446597360518e+20*cos(theta)**2 - 2.13562357620778e+18)*sin(14*phi)

@torch.jit.script
def Yl24_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.75573370217054e-18*(1.0 - cos(theta)**2)**6.5*(2.98763476224155e+22*cos(theta)**11 - 3.4961683387933e+22*cos(theta)**9 + 1.39846733551732e+22*cos(theta)**7 - 2.2765747322375e+21*cos(theta)**5 + 1.38815532453506e+20*cos(theta)**3 - 2.13562357620778e+18*cos(theta))*sin(13*phi)

@torch.jit.script
def Yl24_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.00209527253683e-16*(1.0 - cos(theta)**2)**6*(2.48969563520129e+21*cos(theta)**12 - 3.4961683387933e+21*cos(theta)**10 + 1.74808416939665e+21*cos(theta)**8 - 3.79429122039583e+20*cos(theta)**6 + 3.47038831133765e+19*cos(theta)**4 - 1.06781178810389e+18*cos(theta)**2 + 4.80996300947699e+15)*sin(12*phi)

@torch.jit.script
def Yl24_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.16786353281895e-15*(1.0 - cos(theta)**2)**5.5*(1.91515048861638e+20*cos(theta)**13 - 3.17833485344846e+20*cos(theta)**11 + 1.94231574377406e+20*cos(theta)**9 - 5.4204160291369e+19*cos(theta)**7 + 6.9407766226753e+18*cos(theta)**5 - 3.55937262701297e+17*cos(theta)**3 + 4.80996300947699e+15*cos(theta))*sin(11*phi)

@torch.jit.script
def Yl24_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.79877049408895e-14*(1.0 - cos(theta)**2)**5*(1.36796463472598e+19*cos(theta)**14 - 2.64861237787371e+19*cos(theta)**12 + 1.94231574377406e+19*cos(theta)**10 - 6.77552003642113e+18*cos(theta)**8 + 1.15679610377922e+18*cos(theta)**6 - 8.89843156753244e+16*cos(theta)**4 + 2.4049815047385e+15*cos(theta)**2 - 9816251039748.96)*sin(10*phi)

@torch.jit.script
def Yl24_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.08371495837322e-12*(1.0 - cos(theta)**2)**4.5*(9.11976423150656e+17*cos(theta)**15 - 2.03739413682593e+18*cos(theta)**13 + 1.76574158524914e+18*cos(theta)**11 - 7.52835559602347e+17*cos(theta)**9 + 1.65256586254174e+17*cos(theta)**7 - 1.77968631350649e+16*cos(theta)**5 + 801660501579499.0*cos(theta)**3 - 9816251039748.96*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl24_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.49018738774613e-11*(1.0 - cos(theta)**2)**4*(5.6998526446916e+16*cos(theta)**16 - 1.45528152630424e+17*cos(theta)**14 + 1.47145132104095e+17*cos(theta)**12 - 7.52835559602347e+16*cos(theta)**10 + 2.06570732817717e+16*cos(theta)**8 - 2.96614385584415e+15*cos(theta)**6 + 200415125394875.0*cos(theta)**4 - 4908125519874.48*cos(theta)**2 + 18591384544.9791)*sin(8*phi)

@torch.jit.script
def Yl24_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.80806514683927e-10*(1.0 - cos(theta)**2)**3.5*(3.35285449687741e+15*cos(theta)**17 - 9.70187684202825e+15*cos(theta)**15 + 1.13188563156996e+16*cos(theta)**13 - 6.84395963274861e+15*cos(theta)**11 + 2.2952303646413e+15*cos(theta)**9 - 423734836549164.0*cos(theta)**7 + 40083025078974.9*cos(theta)**5 - 1636041839958.16*cos(theta)**3 + 18591384544.9791*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl24_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.37198252096958e-8*(1.0 - cos(theta)**2)**3*(186269694270967.0*cos(theta)**18 - 606367302626766.0*cos(theta)**16 + 808489736835688.0*cos(theta)**14 - 570329969395718.0*cos(theta)**12 + 229523036464130.0*cos(theta)**10 - 52966854568645.4*cos(theta)**8 + 6680504179829.16*cos(theta)**6 - 409010459989.54*cos(theta)**4 + 9295692272.48955*cos(theta)**2 - 33317893.4497833)*sin(6*phi)

@torch.jit.script
def Yl24_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.27556337379121e-7*(1.0 - cos(theta)**2)**2.5*(9803668119524.59*cos(theta)**19 - 35668664860398.0*cos(theta)**17 + 53899315789045.8*cos(theta)**15 - 43871536107362.9*cos(theta)**13 + 20865730587648.2*cos(theta)**11 - 5885206063182.83*cos(theta)**9 + 954357739975.594*cos(theta)**7 - 81802091997.908*cos(theta)**5 + 3098564090.82985*cos(theta)**3 - 33317893.4497833*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl24_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.88860123286696e-6*(1.0 - cos(theta)**2)**2*(490183405976.23*cos(theta)**20 - 1981592492244.33*cos(theta)**18 + 3368707236815.36*cos(theta)**16 - 3133681150525.92*cos(theta)**14 + 1738810882304.02*cos(theta)**12 - 588520606318.283*cos(theta)**10 + 119294717496.949*cos(theta)**8 - 13633681999.6513*cos(theta)**6 + 774641022.707462*cos(theta)**4 - 16658946.7248917*cos(theta)**2 + 57444.6438789368)*sin(4*phi)

@torch.jit.script
def Yl24_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000191288413903665*(1.0 - cos(theta)**2)**1.5*(23342066951.249*cos(theta)**21 - 104294341697.07*cos(theta)**19 + 198159249224.433*cos(theta)**17 - 208912076701.728*cos(theta)**15 + 133754683254.155*cos(theta)**13 - 53501873301.6621*cos(theta)**11 + 13254968610.7721*cos(theta)**9 - 1947668857.09305*cos(theta)**7 + 154928204.541492*cos(theta)**5 - 5552982.24163055*cos(theta)**3 + 57444.6438789368*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl24_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00466210326274582*(1.0 - cos(theta)**2)*(1061003043.23859*cos(theta)**22 - 5214717084.85351*cos(theta)**20 + 11008847179.1352*cos(theta)**18 - 13057004793.858*cos(theta)**16 + 9553905946.72537*cos(theta)**14 - 4458489441.80517*cos(theta)**12 + 1325496861.07721*cos(theta)**10 - 243458607.136631*cos(theta)**8 + 25821367.4235821*cos(theta)**6 - 1388245.56040764*cos(theta)**4 + 28722.3219394684*cos(theta)**2 - 96.7081546783447)*sin(2*phi)

@torch.jit.script
def Yl24_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.114007252777348*(1.0 - cos(theta)**2)**0.5*(46130567.0973301*cos(theta)**23 - 248319861.1835*cos(theta)**21 + 579413009.428167*cos(theta)**19 - 768059105.521059*cos(theta)**17 + 636927063.115025*cos(theta)**15 - 342960726.292706*cos(theta)**13 + 120499714.643383*cos(theta)**11 - 27050956.3485146*cos(theta)**9 + 3688766.77479744*cos(theta)**7 - 277649.112081528*cos(theta)**5 + 9574.10731315613*cos(theta)**3 - 96.7081546783447*cos(theta))*sin(phi)

@torch.jit.script
def Yl24_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3795514.54325524*cos(theta)**24 - 22288553.488052*cos(theta)**22 + 57207287.2860002*cos(theta)**20 - 84258795.2274422*cos(theta)**18 + 78607290.669504*cos(theta)**16 - 48373717.3350794*cos(theta)**14 + 19828866.1148298*cos(theta)**12 - 5341653.72889294*cos(theta)**10 + 910509.158334023*cos(theta)**8 - 91377.2632019808*cos(theta)**6 + 4726.4101656197*cos(theta)**4 - 95.4830336488828*cos(theta)**2 + 0.318276778829609

@torch.jit.script
def Yl24_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.114007252777348*(1.0 - cos(theta)**2)**0.5*(46130567.0973301*cos(theta)**23 - 248319861.1835*cos(theta)**21 + 579413009.428167*cos(theta)**19 - 768059105.521059*cos(theta)**17 + 636927063.115025*cos(theta)**15 - 342960726.292706*cos(theta)**13 + 120499714.643383*cos(theta)**11 - 27050956.3485146*cos(theta)**9 + 3688766.77479744*cos(theta)**7 - 277649.112081528*cos(theta)**5 + 9574.10731315613*cos(theta)**3 - 96.7081546783447*cos(theta))*cos(phi)

@torch.jit.script
def Yl24_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00466210326274582*(1.0 - cos(theta)**2)*(1061003043.23859*cos(theta)**22 - 5214717084.85351*cos(theta)**20 + 11008847179.1352*cos(theta)**18 - 13057004793.858*cos(theta)**16 + 9553905946.72537*cos(theta)**14 - 4458489441.80517*cos(theta)**12 + 1325496861.07721*cos(theta)**10 - 243458607.136631*cos(theta)**8 + 25821367.4235821*cos(theta)**6 - 1388245.56040764*cos(theta)**4 + 28722.3219394684*cos(theta)**2 - 96.7081546783447)*cos(2*phi)

@torch.jit.script
def Yl24_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000191288413903665*(1.0 - cos(theta)**2)**1.5*(23342066951.249*cos(theta)**21 - 104294341697.07*cos(theta)**19 + 198159249224.433*cos(theta)**17 - 208912076701.728*cos(theta)**15 + 133754683254.155*cos(theta)**13 - 53501873301.6621*cos(theta)**11 + 13254968610.7721*cos(theta)**9 - 1947668857.09305*cos(theta)**7 + 154928204.541492*cos(theta)**5 - 5552982.24163055*cos(theta)**3 + 57444.6438789368*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl24_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.88860123286696e-6*(1.0 - cos(theta)**2)**2*(490183405976.23*cos(theta)**20 - 1981592492244.33*cos(theta)**18 + 3368707236815.36*cos(theta)**16 - 3133681150525.92*cos(theta)**14 + 1738810882304.02*cos(theta)**12 - 588520606318.283*cos(theta)**10 + 119294717496.949*cos(theta)**8 - 13633681999.6513*cos(theta)**6 + 774641022.707462*cos(theta)**4 - 16658946.7248917*cos(theta)**2 + 57444.6438789368)*cos(4*phi)

@torch.jit.script
def Yl24_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.27556337379121e-7*(1.0 - cos(theta)**2)**2.5*(9803668119524.59*cos(theta)**19 - 35668664860398.0*cos(theta)**17 + 53899315789045.8*cos(theta)**15 - 43871536107362.9*cos(theta)**13 + 20865730587648.2*cos(theta)**11 - 5885206063182.83*cos(theta)**9 + 954357739975.594*cos(theta)**7 - 81802091997.908*cos(theta)**5 + 3098564090.82985*cos(theta)**3 - 33317893.4497833*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl24_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.37198252096958e-8*(1.0 - cos(theta)**2)**3*(186269694270967.0*cos(theta)**18 - 606367302626766.0*cos(theta)**16 + 808489736835688.0*cos(theta)**14 - 570329969395718.0*cos(theta)**12 + 229523036464130.0*cos(theta)**10 - 52966854568645.4*cos(theta)**8 + 6680504179829.16*cos(theta)**6 - 409010459989.54*cos(theta)**4 + 9295692272.48955*cos(theta)**2 - 33317893.4497833)*cos(6*phi)

@torch.jit.script
def Yl24_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.80806514683927e-10*(1.0 - cos(theta)**2)**3.5*(3.35285449687741e+15*cos(theta)**17 - 9.70187684202825e+15*cos(theta)**15 + 1.13188563156996e+16*cos(theta)**13 - 6.84395963274861e+15*cos(theta)**11 + 2.2952303646413e+15*cos(theta)**9 - 423734836549164.0*cos(theta)**7 + 40083025078974.9*cos(theta)**5 - 1636041839958.16*cos(theta)**3 + 18591384544.9791*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl24_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.49018738774613e-11*(1.0 - cos(theta)**2)**4*(5.6998526446916e+16*cos(theta)**16 - 1.45528152630424e+17*cos(theta)**14 + 1.47145132104095e+17*cos(theta)**12 - 7.52835559602347e+16*cos(theta)**10 + 2.06570732817717e+16*cos(theta)**8 - 2.96614385584415e+15*cos(theta)**6 + 200415125394875.0*cos(theta)**4 - 4908125519874.48*cos(theta)**2 + 18591384544.9791)*cos(8*phi)

@torch.jit.script
def Yl24_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.08371495837322e-12*(1.0 - cos(theta)**2)**4.5*(9.11976423150656e+17*cos(theta)**15 - 2.03739413682593e+18*cos(theta)**13 + 1.76574158524914e+18*cos(theta)**11 - 7.52835559602347e+17*cos(theta)**9 + 1.65256586254174e+17*cos(theta)**7 - 1.77968631350649e+16*cos(theta)**5 + 801660501579499.0*cos(theta)**3 - 9816251039748.96*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl24_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.79877049408895e-14*(1.0 - cos(theta)**2)**5*(1.36796463472598e+19*cos(theta)**14 - 2.64861237787371e+19*cos(theta)**12 + 1.94231574377406e+19*cos(theta)**10 - 6.77552003642113e+18*cos(theta)**8 + 1.15679610377922e+18*cos(theta)**6 - 8.89843156753244e+16*cos(theta)**4 + 2.4049815047385e+15*cos(theta)**2 - 9816251039748.96)*cos(10*phi)

@torch.jit.script
def Yl24_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.16786353281895e-15*(1.0 - cos(theta)**2)**5.5*(1.91515048861638e+20*cos(theta)**13 - 3.17833485344846e+20*cos(theta)**11 + 1.94231574377406e+20*cos(theta)**9 - 5.4204160291369e+19*cos(theta)**7 + 6.9407766226753e+18*cos(theta)**5 - 3.55937262701297e+17*cos(theta)**3 + 4.80996300947699e+15*cos(theta))*cos(11*phi)

@torch.jit.script
def Yl24_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.00209527253683e-16*(1.0 - cos(theta)**2)**6*(2.48969563520129e+21*cos(theta)**12 - 3.4961683387933e+21*cos(theta)**10 + 1.74808416939665e+21*cos(theta)**8 - 3.79429122039583e+20*cos(theta)**6 + 3.47038831133765e+19*cos(theta)**4 - 1.06781178810389e+18*cos(theta)**2 + 4.80996300947699e+15)*cos(12*phi)

@torch.jit.script
def Yl24_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.75573370217054e-18*(1.0 - cos(theta)**2)**6.5*(2.98763476224155e+22*cos(theta)**11 - 3.4961683387933e+22*cos(theta)**9 + 1.39846733551732e+22*cos(theta)**7 - 2.2765747322375e+21*cos(theta)**5 + 1.38815532453506e+20*cos(theta)**3 - 2.13562357620778e+18*cos(theta))*cos(13*phi)

@torch.jit.script
def Yl24_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.32610538861376e-19*(1.0 - cos(theta)**2)**7*(3.2863982384657e+23*cos(theta)**10 - 3.14655150491397e+23*cos(theta)**8 + 9.78927134862124e+22*cos(theta)**6 - 1.13828736611875e+22*cos(theta)**4 + 4.16446597360518e+20*cos(theta)**2 - 2.13562357620778e+18)*cos(14*phi)

@torch.jit.script
def Yl24_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.1778692495173e-20*(1.0 - cos(theta)**2)**7.5*(3.2863982384657e+24*cos(theta)**9 - 2.51724120393118e+24*cos(theta)**7 + 5.87356280917274e+23*cos(theta)**5 - 4.553149464475e+22*cos(theta)**3 + 8.32893194721036e+20*cos(theta))*cos(15*phi)

@torch.jit.script
def Yl24_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.2079160239131e-22*(1.0 - cos(theta)**2)**8*(2.95775841461913e+25*cos(theta)**8 - 1.76206884275182e+25*cos(theta)**6 + 2.93678140458637e+24*cos(theta)**4 - 1.3659448393425e+23*cos(theta)**2 + 8.32893194721036e+20)*cos(16*phi)

@torch.jit.script
def Yl24_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.42774820132609e-23*(1.0 - cos(theta)**2)**8.5*(2.36620673169531e+26*cos(theta)**7 - 1.05724130565109e+26*cos(theta)**5 + 1.17471256183455e+25*cos(theta)**3 - 2.731889678685e+23*cos(theta))*cos(17*phi)

@torch.jit.script
def Yl24_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.99910334761708e-24*(1.0 - cos(theta)**2)**9*(1.65634471218671e+27*cos(theta)**6 - 5.28620652825547e+26*cos(theta)**4 + 3.52413768550365e+25*cos(theta)**2 - 2.731889678685e+23)*cos(18*phi)

@torch.jit.script
def Yl24_m19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.24458738133901e-25*(1.0 - cos(theta)**2)**9.5*(9.93806827312028e+27*cos(theta)**5 - 2.11448261130219e+27*cos(theta)**3 + 7.04827537100729e+25*cos(theta))*cos(19*phi)

@torch.jit.script
def Yl24_m20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.39100641322249e-27*(1.0 - cos(theta)**2)**10*(4.96903413656014e+28*cos(theta)**4 - 6.34344783390656e+27*cos(theta)**2 + 7.04827537100729e+25)*cos(20*phi)

@torch.jit.script
def Yl24_m21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.25428691320073e-28*(1.0 - cos(theta)**2)**10.5*(1.98761365462406e+29*cos(theta)**3 - 1.26868956678131e+28*cos(theta))*cos(21*phi)

@torch.jit.script
def Yl24_m22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.3240025801011e-29*(1.0 - cos(theta)**2)**11*(5.96284096387217e+29*cos(theta)**2 - 1.26868956678131e+28)*cos(22*phi)

@torch.jit.script
def Yl24_m23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.54873704743938*(1.0 - cos(theta)**2)**11.5*cos(23*phi)*cos(theta)

@torch.jit.script
def Yl24_m24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.9452287742978*(1.0 - cos(theta)**2)**12*cos(24*phi)

@torch.jit.script
def Yl25_m_minus_25(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.954634267390256*(1.0 - cos(theta)**2)**12.5*sin(25*phi)

@torch.jit.script
def Yl25_m_minus_24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.75028364024702*(1.0 - cos(theta)**2)**12*sin(24*phi)*cos(theta)

@torch.jit.script
def Yl25_m_minus_23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.14355157834306e-30*(1.0 - cos(theta)**2)**11.5*(2.92179207229736e+31*cos(theta)**2 - 5.96284096387217e+29)*sin(23*phi)

@torch.jit.script
def Yl25_m_minus_22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.37226189401167e-29*(1.0 - cos(theta)**2)**11*(9.73930690765788e+30*cos(theta)**3 - 5.96284096387217e+29*cos(theta))*sin(22*phi)

@torch.jit.script
def Yl25_m_minus_21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.88155071332724e-28*(1.0 - cos(theta)**2)**10.5*(2.43482672691447e+30*cos(theta)**4 - 2.98142048193609e+29*cos(theta)**2 + 3.17172391695328e+27)*sin(21*phi)

@torch.jit.script
def Yl25_m_minus_20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.85351294016536e-27*(1.0 - cos(theta)**2)**10*(4.86965345382894e+29*cos(theta)**5 - 9.93806827312028e+28*cos(theta)**3 + 3.17172391695328e+27*cos(theta))*sin(20*phi)

@torch.jit.script
def Yl25_m_minus_19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.68880021638437e-26*(1.0 - cos(theta)**2)**9.5*(8.1160890897149e+28*cos(theta)**6 - 2.48451706828007e+28*cos(theta)**4 + 1.58586195847664e+27*cos(theta)**2 - 1.17471256183455e+25)*sin(19*phi)

@torch.jit.script
def Yl25_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.22881098367386e-25*(1.0 - cos(theta)**2)**9*(1.1594412985307e+28*cos(theta)**7 - 4.96903413656014e+27*cos(theta)**5 + 5.28620652825547e+26*cos(theta)**3 - 1.17471256183455e+25*cos(theta))*sin(18*phi)

@torch.jit.script
def Yl25_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.52621707468272e-23*(1.0 - cos(theta)**2)**8.5*(1.44930162316337e+27*cos(theta)**8 - 8.28172356093357e+26*cos(theta)**6 + 1.32155163206387e+26*cos(theta)**4 - 5.87356280917274e+24*cos(theta)**2 + 3.41486209835625e+22)*sin(17*phi)

@torch.jit.script
def Yl25_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.96730513315039e-22*(1.0 - cos(theta)**2)**8*(1.61033513684819e+26*cos(theta)**9 - 1.18310336584765e+26*cos(theta)**7 + 2.64310326412774e+25*cos(theta)**5 - 1.95785426972425e+24*cos(theta)**3 + 3.41486209835625e+22*cos(theta))*sin(16*phi)

@torch.jit.script
def Yl25_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.00833495972093e-21*(1.0 - cos(theta)**2)**7.5*(1.61033513684819e+25*cos(theta)**10 - 1.47887920730957e+25*cos(theta)**8 + 4.40517210687956e+24*cos(theta)**6 - 4.89463567431062e+23*cos(theta)**4 + 1.70743104917812e+22*cos(theta)**2 - 8.32893194721036e+19)*sin(15*phi)

@torch.jit.script
def Yl25_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.26031897370507e-19*(1.0 - cos(theta)**2)**7*(1.46394103349836e+24*cos(theta)**11 - 1.64319911923285e+24*cos(theta)**9 + 6.29310300982794e+23*cos(theta)**7 - 9.78927134862124e+22*cos(theta)**5 + 5.69143683059375e+21*cos(theta)**3 - 8.32893194721036e+19*cos(theta))*sin(14*phi)

@torch.jit.script
def Yl25_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.72648680988027e-18*(1.0 - cos(theta)**2)**6.5*(1.21995086124863e+23*cos(theta)**12 - 1.64319911923285e+23*cos(theta)**10 + 7.86637876228493e+22*cos(theta)**8 - 1.63154522477021e+22*cos(theta)**6 + 1.42285920764844e+21*cos(theta)**4 - 4.16446597360518e+19*cos(theta)**2 + 1.77968631350649e+17)*sin(13*phi)

@torch.jit.script
def Yl25_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.05991978517773e-17*(1.0 - cos(theta)**2)**6*(9.38423739422025e+21*cos(theta)**13 - 1.49381738112077e+22*cos(theta)**11 + 8.74042084698325e+21*cos(theta)**9 - 2.33077889252887e+21*cos(theta)**7 + 2.84571841529687e+20*cos(theta)**5 - 1.38815532453506e+19*cos(theta)**3 + 1.77968631350649e+17*cos(theta))*sin(12*phi)

@torch.jit.script
def Yl25_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.37921431263761e-15*(1.0 - cos(theta)**2)**5.5*(6.70302671015732e+20*cos(theta)**14 - 1.24484781760064e+21*cos(theta)**12 + 8.74042084698325e+20*cos(theta)**10 - 2.91347361566108e+20*cos(theta)**8 + 4.74286402549479e+19*cos(theta)**6 - 3.47038831133765e+18*cos(theta)**4 + 8.89843156753244e+16*cos(theta)**2 - 343568786391214.0)*sin(11*phi)

@torch.jit.script
def Yl25_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.20500443821783e-14*(1.0 - cos(theta)**2)**5*(4.46868447343821e+19*cos(theta)**15 - 9.57575244308188e+19*cos(theta)**13 + 7.94583713362114e+19*cos(theta)**11 - 3.23719290629009e+19*cos(theta)**9 + 6.77552003642113e+18*cos(theta)**7 - 6.9407766226753e+17*cos(theta)**5 + 2.96614385584414e+16*cos(theta)**3 - 343568786391214.0*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl25_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.58442478467402e-13*(1.0 - cos(theta)**2)**4.5*(2.79292779589888e+18*cos(theta)**16 - 6.83982317362992e+18*cos(theta)**14 + 6.62153094468428e+18*cos(theta)**12 - 3.23719290629009e+18*cos(theta)**10 + 8.46940004552641e+17*cos(theta)**8 - 1.15679610377922e+17*cos(theta)**6 + 7.41535963961036e+15*cos(theta)**4 - 171784393195607.0*cos(theta)**2 + 613515689984.31)*sin(9*phi)

@torch.jit.script
def Yl25_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.82341938685839e-11*(1.0 - cos(theta)**2)**4*(1.64289870346993e+17*cos(theta)**17 - 4.55988211575328e+17*cos(theta)**15 + 5.09348534206483e+17*cos(theta)**13 - 2.9429026420819e+17*cos(theta)**11 + 9.41044449502934e+16*cos(theta)**9 - 1.65256586254174e+16*cos(theta)**7 + 1.48307192792207e+15*cos(theta)**5 - 57261464398535.6*cos(theta)**3 + 613515689984.31*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl25_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.44405873797859e-10*(1.0 - cos(theta)**2)**3.5*(9.12721501927739e+15*cos(theta)**18 - 2.8499263223458e+16*cos(theta)**16 + 3.63820381576059e+16*cos(theta)**14 - 2.45241886840159e+16*cos(theta)**12 + 9.41044449502934e+15*cos(theta)**10 - 2.06570732817717e+15*cos(theta)**8 + 247178654653679.0*cos(theta)**6 - 14315366099633.9*cos(theta)**4 + 306757844992.155*cos(theta)**2 - 1032854696.94328)*sin(7*phi)

@torch.jit.script
def Yl25_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.09580071657648e-8*(1.0 - cos(theta)**2)**3*(480379737856705.0*cos(theta)**19 - 1.67642724843871e+15*cos(theta)**17 + 2.42546921050706e+15*cos(theta)**15 - 1.8864760526166e+15*cos(theta)**13 + 855494954093576.0*cos(theta)**11 - 229523036464130.0*cos(theta)**9 + 35311236379097.0*cos(theta)**7 - 2863073219926.78*cos(theta)**5 + 102252614997.385*cos(theta)**3 - 1032854696.94328*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl25_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.72852178015624e-7*(1.0 - cos(theta)**2)**2.5*(24018986892835.3*cos(theta)**20 - 93134847135483.6*cos(theta)**18 + 151591825656691.0*cos(theta)**16 - 134748289472615.0*cos(theta)**14 + 71291246174464.7*cos(theta)**12 - 22952303646413.0*cos(theta)**10 + 4413904547387.12*cos(theta)**8 - 477178869987.797*cos(theta)**6 + 25563153749.3463*cos(theta)**4 - 516427348.471642*cos(theta)**2 + 1665894.67248917)*sin(5*phi)

@torch.jit.script
def Yl25_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.84853531495298e-6*(1.0 - cos(theta)**2)**2*(1143761280611.2*cos(theta)**21 - 4901834059762.3*cos(theta)**19 + 8917166215099.5*cos(theta)**17 - 8983219298174.31*cos(theta)**15 + 5483942013420.36*cos(theta)**13 - 2086573058764.82*cos(theta)**11 + 490433838598.569*cos(theta)**9 - 68168409998.2567*cos(theta)**7 + 5112630749.86925*cos(theta)**5 - 172142449.490547*cos(theta)**3 + 1665894.67248917*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl25_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000172984837897952*(1.0 - cos(theta)**2)**1.5*(51989149118.691*cos(theta)**22 - 245091702988.115*cos(theta)**20 + 495398123061.083*cos(theta)**18 - 561451206135.894*cos(theta)**16 + 391710143815.74*cos(theta)**14 - 173881088230.402*cos(theta)**12 + 49043383859.8569*cos(theta)**10 - 8521051249.78209*cos(theta)**8 + 852105124.978209*cos(theta)**6 - 43035612.3726368*cos(theta)**4 + 832947.336244583*cos(theta)**2 - 2611.12017631531)*sin(3*phi)

@torch.jit.script
def Yl25_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00438986305798052*(1.0 - cos(theta)**2)*(2260397787.76917*cos(theta)**23 - 11671033475.6245*cos(theta)**21 + 26073585424.2675*cos(theta)**19 - 33026541537.4055*cos(theta)**17 + 26114009587.716*cos(theta)**15 - 13375468325.4155*cos(theta)**13 + 4458489441.80517*cos(theta)**11 - 946783472.198009*cos(theta)**9 + 121729303.568316*cos(theta)**7 - 8607122.47452736*cos(theta)**5 + 277649.112081528*cos(theta)**3 - 2611.12017631531*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl25_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.11174766972402*(1.0 - cos(theta)**2)**0.5*(94183241.1570489*cos(theta)**24 - 530501521.619296*cos(theta)**22 + 1303679271.21338*cos(theta)**20 - 1834807863.1892*cos(theta)**18 + 1632125599.23225*cos(theta)**16 - 955390594.672537*cos(theta)**14 + 371540786.817098*cos(theta)**12 - 94678347.2198009*cos(theta)**10 + 15216162.9460394*cos(theta)**8 - 1434520.41242123*cos(theta)**6 + 69412.2780203819*cos(theta)**4 - 1305.56008815765*cos(theta)**2 + 4.02950644493103)*sin(phi)

@torch.jit.script
def Yl25_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7589510.72884222*cos(theta)**25 - 46466392.2174014*cos(theta)**23 + 125063800.329814*cos(theta)**21 - 194543689.401933*cos(theta)**19 + 193412621.440294*cos(theta)**17 - 128312763.492098*cos(theta)**15 + 57576240.0285053*cos(theta)**13 - 17339562.6340672*cos(theta)**11 + 3405985.51740607*cos(theta)**9 - 412846.729382553*cos(theta)**7 + 27967.0365065601*cos(theta)**5 - 876.709608356115*cos(theta)**3 + 8.11768155885292*cos(theta)

@torch.jit.script
def Yl25_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.11174766972402*(1.0 - cos(theta)**2)**0.5*(94183241.1570489*cos(theta)**24 - 530501521.619296*cos(theta)**22 + 1303679271.21338*cos(theta)**20 - 1834807863.1892*cos(theta)**18 + 1632125599.23225*cos(theta)**16 - 955390594.672537*cos(theta)**14 + 371540786.817098*cos(theta)**12 - 94678347.2198009*cos(theta)**10 + 15216162.9460394*cos(theta)**8 - 1434520.41242123*cos(theta)**6 + 69412.2780203819*cos(theta)**4 - 1305.56008815765*cos(theta)**2 + 4.02950644493103)*cos(phi)

@torch.jit.script
def Yl25_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00438986305798052*(1.0 - cos(theta)**2)*(2260397787.76917*cos(theta)**23 - 11671033475.6245*cos(theta)**21 + 26073585424.2675*cos(theta)**19 - 33026541537.4055*cos(theta)**17 + 26114009587.716*cos(theta)**15 - 13375468325.4155*cos(theta)**13 + 4458489441.80517*cos(theta)**11 - 946783472.198009*cos(theta)**9 + 121729303.568316*cos(theta)**7 - 8607122.47452736*cos(theta)**5 + 277649.112081528*cos(theta)**3 - 2611.12017631531*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl25_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000172984837897952*(1.0 - cos(theta)**2)**1.5*(51989149118.691*cos(theta)**22 - 245091702988.115*cos(theta)**20 + 495398123061.083*cos(theta)**18 - 561451206135.894*cos(theta)**16 + 391710143815.74*cos(theta)**14 - 173881088230.402*cos(theta)**12 + 49043383859.8569*cos(theta)**10 - 8521051249.78209*cos(theta)**8 + 852105124.978209*cos(theta)**6 - 43035612.3726368*cos(theta)**4 + 832947.336244583*cos(theta)**2 - 2611.12017631531)*cos(3*phi)

@torch.jit.script
def Yl25_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.84853531495298e-6*(1.0 - cos(theta)**2)**2*(1143761280611.2*cos(theta)**21 - 4901834059762.3*cos(theta)**19 + 8917166215099.5*cos(theta)**17 - 8983219298174.31*cos(theta)**15 + 5483942013420.36*cos(theta)**13 - 2086573058764.82*cos(theta)**11 + 490433838598.569*cos(theta)**9 - 68168409998.2567*cos(theta)**7 + 5112630749.86925*cos(theta)**5 - 172142449.490547*cos(theta)**3 + 1665894.67248917*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl25_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.72852178015624e-7*(1.0 - cos(theta)**2)**2.5*(24018986892835.3*cos(theta)**20 - 93134847135483.6*cos(theta)**18 + 151591825656691.0*cos(theta)**16 - 134748289472615.0*cos(theta)**14 + 71291246174464.7*cos(theta)**12 - 22952303646413.0*cos(theta)**10 + 4413904547387.12*cos(theta)**8 - 477178869987.797*cos(theta)**6 + 25563153749.3463*cos(theta)**4 - 516427348.471642*cos(theta)**2 + 1665894.67248917)*cos(5*phi)

@torch.jit.script
def Yl25_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.09580071657648e-8*(1.0 - cos(theta)**2)**3*(480379737856705.0*cos(theta)**19 - 1.67642724843871e+15*cos(theta)**17 + 2.42546921050706e+15*cos(theta)**15 - 1.8864760526166e+15*cos(theta)**13 + 855494954093576.0*cos(theta)**11 - 229523036464130.0*cos(theta)**9 + 35311236379097.0*cos(theta)**7 - 2863073219926.78*cos(theta)**5 + 102252614997.385*cos(theta)**3 - 1032854696.94328*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl25_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.44405873797859e-10*(1.0 - cos(theta)**2)**3.5*(9.12721501927739e+15*cos(theta)**18 - 2.8499263223458e+16*cos(theta)**16 + 3.63820381576059e+16*cos(theta)**14 - 2.45241886840159e+16*cos(theta)**12 + 9.41044449502934e+15*cos(theta)**10 - 2.06570732817717e+15*cos(theta)**8 + 247178654653679.0*cos(theta)**6 - 14315366099633.9*cos(theta)**4 + 306757844992.155*cos(theta)**2 - 1032854696.94328)*cos(7*phi)

@torch.jit.script
def Yl25_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.82341938685839e-11*(1.0 - cos(theta)**2)**4*(1.64289870346993e+17*cos(theta)**17 - 4.55988211575328e+17*cos(theta)**15 + 5.09348534206483e+17*cos(theta)**13 - 2.9429026420819e+17*cos(theta)**11 + 9.41044449502934e+16*cos(theta)**9 - 1.65256586254174e+16*cos(theta)**7 + 1.48307192792207e+15*cos(theta)**5 - 57261464398535.6*cos(theta)**3 + 613515689984.31*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl25_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.58442478467402e-13*(1.0 - cos(theta)**2)**4.5*(2.79292779589888e+18*cos(theta)**16 - 6.83982317362992e+18*cos(theta)**14 + 6.62153094468428e+18*cos(theta)**12 - 3.23719290629009e+18*cos(theta)**10 + 8.46940004552641e+17*cos(theta)**8 - 1.15679610377922e+17*cos(theta)**6 + 7.41535963961036e+15*cos(theta)**4 - 171784393195607.0*cos(theta)**2 + 613515689984.31)*cos(9*phi)

@torch.jit.script
def Yl25_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.20500443821783e-14*(1.0 - cos(theta)**2)**5*(4.46868447343821e+19*cos(theta)**15 - 9.57575244308188e+19*cos(theta)**13 + 7.94583713362114e+19*cos(theta)**11 - 3.23719290629009e+19*cos(theta)**9 + 6.77552003642113e+18*cos(theta)**7 - 6.9407766226753e+17*cos(theta)**5 + 2.96614385584414e+16*cos(theta)**3 - 343568786391214.0*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl25_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.37921431263761e-15*(1.0 - cos(theta)**2)**5.5*(6.70302671015732e+20*cos(theta)**14 - 1.24484781760064e+21*cos(theta)**12 + 8.74042084698325e+20*cos(theta)**10 - 2.91347361566108e+20*cos(theta)**8 + 4.74286402549479e+19*cos(theta)**6 - 3.47038831133765e+18*cos(theta)**4 + 8.89843156753244e+16*cos(theta)**2 - 343568786391214.0)*cos(11*phi)

@torch.jit.script
def Yl25_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.05991978517773e-17*(1.0 - cos(theta)**2)**6*(9.38423739422025e+21*cos(theta)**13 - 1.49381738112077e+22*cos(theta)**11 + 8.74042084698325e+21*cos(theta)**9 - 2.33077889252887e+21*cos(theta)**7 + 2.84571841529687e+20*cos(theta)**5 - 1.38815532453506e+19*cos(theta)**3 + 1.77968631350649e+17*cos(theta))*cos(12*phi)

@torch.jit.script
def Yl25_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.72648680988027e-18*(1.0 - cos(theta)**2)**6.5*(1.21995086124863e+23*cos(theta)**12 - 1.64319911923285e+23*cos(theta)**10 + 7.86637876228493e+22*cos(theta)**8 - 1.63154522477021e+22*cos(theta)**6 + 1.42285920764844e+21*cos(theta)**4 - 4.16446597360518e+19*cos(theta)**2 + 1.77968631350649e+17)*cos(13*phi)

@torch.jit.script
def Yl25_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.26031897370507e-19*(1.0 - cos(theta)**2)**7*(1.46394103349836e+24*cos(theta)**11 - 1.64319911923285e+24*cos(theta)**9 + 6.29310300982794e+23*cos(theta)**7 - 9.78927134862124e+22*cos(theta)**5 + 5.69143683059375e+21*cos(theta)**3 - 8.32893194721036e+19*cos(theta))*cos(14*phi)

@torch.jit.script
def Yl25_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.00833495972093e-21*(1.0 - cos(theta)**2)**7.5*(1.61033513684819e+25*cos(theta)**10 - 1.47887920730957e+25*cos(theta)**8 + 4.40517210687956e+24*cos(theta)**6 - 4.89463567431062e+23*cos(theta)**4 + 1.70743104917812e+22*cos(theta)**2 - 8.32893194721036e+19)*cos(15*phi)

@torch.jit.script
def Yl25_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.96730513315039e-22*(1.0 - cos(theta)**2)**8*(1.61033513684819e+26*cos(theta)**9 - 1.18310336584765e+26*cos(theta)**7 + 2.64310326412774e+25*cos(theta)**5 - 1.95785426972425e+24*cos(theta)**3 + 3.41486209835625e+22*cos(theta))*cos(16*phi)

@torch.jit.script
def Yl25_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.52621707468272e-23*(1.0 - cos(theta)**2)**8.5*(1.44930162316337e+27*cos(theta)**8 - 8.28172356093357e+26*cos(theta)**6 + 1.32155163206387e+26*cos(theta)**4 - 5.87356280917274e+24*cos(theta)**2 + 3.41486209835625e+22)*cos(17*phi)

@torch.jit.script
def Yl25_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.22881098367386e-25*(1.0 - cos(theta)**2)**9*(1.1594412985307e+28*cos(theta)**7 - 4.96903413656014e+27*cos(theta)**5 + 5.28620652825547e+26*cos(theta)**3 - 1.17471256183455e+25*cos(theta))*cos(18*phi)

@torch.jit.script
def Yl25_m19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.68880021638437e-26*(1.0 - cos(theta)**2)**9.5*(8.1160890897149e+28*cos(theta)**6 - 2.48451706828007e+28*cos(theta)**4 + 1.58586195847664e+27*cos(theta)**2 - 1.17471256183455e+25)*cos(19*phi)

@torch.jit.script
def Yl25_m20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.85351294016536e-27*(1.0 - cos(theta)**2)**10*(4.86965345382894e+29*cos(theta)**5 - 9.93806827312028e+28*cos(theta)**3 + 3.17172391695328e+27*cos(theta))*cos(20*phi)

@torch.jit.script
def Yl25_m21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.88155071332724e-28*(1.0 - cos(theta)**2)**10.5*(2.43482672691447e+30*cos(theta)**4 - 2.98142048193609e+29*cos(theta)**2 + 3.17172391695328e+27)*cos(21*phi)

@torch.jit.script
def Yl25_m22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.37226189401167e-29*(1.0 - cos(theta)**2)**11*(9.73930690765788e+30*cos(theta)**3 - 5.96284096387217e+29*cos(theta))*cos(22*phi)

@torch.jit.script
def Yl25_m23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.14355157834306e-30*(1.0 - cos(theta)**2)**11.5*(2.92179207229736e+31*cos(theta)**2 - 5.96284096387217e+29)*cos(23*phi)

@torch.jit.script
def Yl25_m24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.75028364024702*(1.0 - cos(theta)**2)**12*cos(24*phi)*cos(theta)

@torch.jit.script
def Yl25_m25(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.954634267390256*(1.0 - cos(theta)**2)**12.5*cos(25*phi)

@torch.jit.script
def Yl26_m_minus_26(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.963769731686801*(1.0 - cos(theta)**2)**13*sin(26*phi)

@torch.jit.script
def Yl26_m_minus_25(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.94984237067386*(1.0 - cos(theta)**2)**12.5*sin(25*phi)*cos(theta)

@torch.jit.script
def Yl26_m_minus_24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.35518790424645e-32*(1.0 - cos(theta)**2)**12*(1.49011395687166e+33*cos(theta)**2 - 2.92179207229736e+31)*sin(24*phi)

@torch.jit.script
def Yl26_m_minus_23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.88450430688934e-31*(1.0 - cos(theta)**2)**11.5*(4.96704652290552e+32*cos(theta)**3 - 2.92179207229736e+31*cos(theta))*sin(23*phi)

@torch.jit.script
def Yl26_m_minus_22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.03830602964508e-30*(1.0 - cos(theta)**2)**11*(1.24176163072638e+32*cos(theta)**4 - 1.46089603614868e+31*cos(theta)**2 + 1.49071024096804e+29)*sin(22*phi)

@torch.jit.script
def Yl26_m_minus_21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.25611679988175e-29*(1.0 - cos(theta)**2)**10.5*(2.48352326145276e+31*cos(theta)**5 - 4.86965345382894e+30*cos(theta)**3 + 1.49071024096804e+29*cos(theta))*sin(21*phi)

@torch.jit.script
def Yl26_m_minus_20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.0505806618571e-27*(1.0 - cos(theta)**2)**10*(4.1392054357546e+30*cos(theta)**6 - 1.21741336345723e+30*cos(theta)**4 + 7.45355120484021e+28*cos(theta)**2 - 5.28620652825547e+26)*sin(20*phi)

@torch.jit.script
def Yl26_m_minus_19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.88519959716718e-26*(1.0 - cos(theta)**2)**9.5*(5.91315062250657e+29*cos(theta)**7 - 2.43482672691447e+29*cos(theta)**5 + 2.48451706828007e+28*cos(theta)**3 - 5.28620652825547e+26*cos(theta))*sin(19*phi)

@torch.jit.script
def Yl26_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.57691474264813e-25*(1.0 - cos(theta)**2)**9*(7.39143827813321e+28*cos(theta)**8 - 4.05804454485745e+28*cos(theta)**6 + 6.21129267070018e+27*cos(theta)**4 - 2.64310326412774e+26*cos(theta)**2 + 1.46839070229319e+24)*sin(18*phi)

@torch.jit.script
def Yl26_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.11797046507269e-24*(1.0 - cos(theta)**2)**8.5*(8.21270919792579e+27*cos(theta)**9 - 5.7972064926535e+27*cos(theta)**7 + 1.24225853414004e+27*cos(theta)**5 - 8.81034421375912e+25*cos(theta)**3 + 1.46839070229319e+24*cos(theta))*sin(17*phi)

@torch.jit.script
def Yl26_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.47601377103699e-22*(1.0 - cos(theta)**2)**8*(8.21270919792579e+26*cos(theta)**10 - 7.24650811581687e+26*cos(theta)**8 + 2.07043089023339e+26*cos(theta)**6 - 2.20258605343978e+25*cos(theta)**4 + 7.34195351146593e+23*cos(theta)**2 - 3.41486209835625e+21)*sin(16*phi)

@torch.jit.script
def Yl26_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.17257134412823e-21*(1.0 - cos(theta)**2)**7.5*(7.46609927084163e+25*cos(theta)**11 - 8.05167568424097e+25*cos(theta)**9 + 2.95775841461913e+25*cos(theta)**7 - 4.40517210687956e+24*cos(theta)**5 + 2.44731783715531e+23*cos(theta)**3 - 3.41486209835625e+21*cos(theta))*sin(15*phi)

@torch.jit.script
def Yl26_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.03710366224851e-20*(1.0 - cos(theta)**2)**7*(6.22174939236802e+24*cos(theta)**12 - 8.05167568424097e+24*cos(theta)**10 + 3.69719801827392e+24*cos(theta)**8 - 7.34195351146593e+23*cos(theta)**6 + 6.11829459288828e+22*cos(theta)**4 - 1.70743104917812e+21*cos(theta)**2 + 6.9407766226753e+18)*sin(14*phi)

@torch.jit.script
def Yl26_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.60470653191418e-18*(1.0 - cos(theta)**2)**6.5*(4.78596107105233e+23*cos(theta)**13 - 7.31970516749179e+23*cos(theta)**11 + 4.10799779808213e+23*cos(theta)**9 - 1.04885050163799e+23*cos(theta)**7 + 1.22365891857766e+22*cos(theta)**5 - 5.69143683059374e+20*cos(theta)**3 + 6.9407766226753e+18*cos(theta))*sin(13*phi)

@torch.jit.script
def Yl26_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.74966044762475e-17*(1.0 - cos(theta)**2)**6*(3.41854362218023e+22*cos(theta)**14 - 6.09975430624316e+22*cos(theta)**12 + 4.10799779808213e+22*cos(theta)**10 - 1.31106312704749e+22*cos(theta)**8 + 2.03943153096276e+21*cos(theta)**6 - 1.42285920764844e+20*cos(theta)**4 + 3.47038831133765e+18*cos(theta)**2 - 1.27120450964749e+16)*sin(12*phi)

@torch.jit.script
def Yl26_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.95219161955017e-16*(1.0 - cos(theta)**2)**5.5*(2.27902908145349e+21*cos(theta)**15 - 4.69211869711012e+21*cos(theta)**13 + 3.73454345280193e+21*cos(theta)**11 - 1.45673680783054e+21*cos(theta)**9 + 2.91347361566108e+20*cos(theta)**7 - 2.84571841529687e+19*cos(theta)**5 + 1.15679610377922e+18*cos(theta)**3 - 1.27120450964749e+16*cos(theta))*sin(11*phi)

@torch.jit.script
def Yl26_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.17816222989798e-14*(1.0 - cos(theta)**2)**5*(1.42439317590843e+20*cos(theta)**16 - 3.35151335507866e+20*cos(theta)**14 + 3.11211954400161e+20*cos(theta)**12 - 1.45673680783054e+20*cos(theta)**10 + 3.64184201957635e+19*cos(theta)**8 - 4.74286402549479e+18*cos(theta)**6 + 2.89199025944804e+17*cos(theta)**4 - 6.35602254823745e+15*cos(theta)**2 + 21473049149450.9)*sin(10*phi)

@torch.jit.script
def Yl26_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.38847576616016e-13*(1.0 - cos(theta)**2)**4.5*(8.37878338769665e+18*cos(theta)**17 - 2.23434223671911e+19*cos(theta)**15 + 2.39393811077047e+19*cos(theta)**13 - 1.32430618893686e+19*cos(theta)**11 + 4.04649113286262e+18*cos(theta)**9 - 6.77552003642113e+17*cos(theta)**7 + 5.78398051889608e+16*cos(theta)**5 - 2.11867418274582e+15*cos(theta)**3 + 21473049149450.9*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl26_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.35249668324814e-11*(1.0 - cos(theta)**2)**4*(4.65487965983147e+17*cos(theta)**18 - 1.39646389794944e+18*cos(theta)**16 + 1.70995579340748e+18*cos(theta)**14 - 1.10358849078071e+18*cos(theta)**12 + 4.04649113286262e+17*cos(theta)**10 - 8.46940004552641e+16*cos(theta)**8 + 9.63996753149347e+15*cos(theta)**6 - 529668545686454.0*cos(theta)**4 + 10736524574725.4*cos(theta)**2 - 34084204999.1283)*sin(8*phi)

@torch.jit.script
def Yl26_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.43757725980871e-10*(1.0 - cos(theta)**2)**3.5*(2.4499366630692e+16*cos(theta)**19 - 8.21449351734965e+16*cos(theta)**17 + 1.13997052893832e+17*cos(theta)**15 - 8.48914223677472e+16*cos(theta)**13 + 3.67862830260238e+16*cos(theta)**11 - 9.41044449502934e+15*cos(theta)**9 + 1.37713821878478e+15*cos(theta)**7 - 105933709137291.0*cos(theta)**5 + 3578841524908.48*cos(theta)**3 - 34084204999.1283*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl26_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.83129588187465e-9*(1.0 - cos(theta)**2)**3*(1.2249683315346e+15*cos(theta)**20 - 4.5636075096387e+15*cos(theta)**18 + 7.1248158058645e+15*cos(theta)**16 - 6.06367302626766e+15*cos(theta)**14 + 3.06552358550198e+15*cos(theta)**12 - 941044449502934.0*cos(theta)**10 + 172142277348098.0*cos(theta)**8 - 17655618189548.5*cos(theta)**6 + 894710381227.119*cos(theta)**4 - 17042102499.5642*cos(theta)**2 + 51642734.8471642)*sin(6*phi)

@torch.jit.script
def Yl26_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.28933354565387e-7*(1.0 - cos(theta)**2)**2.5*(58331825311171.3*cos(theta)**21 - 240189868928353.0*cos(theta)**19 + 419106812109676.0*cos(theta)**17 - 404244868417844.0*cos(theta)**15 + 235809506577076.0*cos(theta)**13 - 85549495409357.6*cos(theta)**11 + 19126919705344.2*cos(theta)**9 - 2522231169935.5*cos(theta)**7 + 178942076245.424*cos(theta)**5 - 5680700833.18806*cos(theta)**3 + 51642734.8471642*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl26_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.97862425042808e-6*(1.0 - cos(theta)**2)**2*(2651446605053.24*cos(theta)**22 - 12009493446417.6*cos(theta)**20 + 23283711783870.9*cos(theta)**18 - 25265304276115.2*cos(theta)**16 + 16843536184076.8*cos(theta)**14 - 7129124617446.47*cos(theta)**12 + 1912691970534.42*cos(theta)**10 - 315278896241.937*cos(theta)**8 + 29823679374.2373*cos(theta)**6 - 1420175208.29701*cos(theta)**4 + 25821367.4235821*cos(theta)**2 - 75722.4851131439)*sin(4*phi)

@torch.jit.script
def Yl26_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000157045611432433*(1.0 - cos(theta)**2)**1.5*(115280287176.228*cos(theta)**23 - 571880640305.601*cos(theta)**21 + 1225458514940.57*cos(theta)**19 - 1486194369183.25*cos(theta)**17 + 1122902412271.79*cos(theta)**15 - 548394201342.036*cos(theta)**13 + 173881088230.402*cos(theta)**11 - 35030988471.3264*cos(theta)**9 + 4260525624.89104*cos(theta)**7 - 284035041.659403*cos(theta)**5 + 8607122.47452736*cos(theta)**3 - 75722.4851131439*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl26_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00414314778312938*(1.0 - cos(theta)**2)*(4803345299.0095*cos(theta)**24 - 25994574559.3455*cos(theta)**22 + 61272925747.0287*cos(theta)**20 - 82566353843.5138*cos(theta)**18 + 70181400766.9868*cos(theta)**16 - 39171014381.574*cos(theta)**14 + 14490090685.8668*cos(theta)**12 - 3503098847.13264*cos(theta)**10 + 532565703.11138*cos(theta)**8 - 47339173.6099005*cos(theta)**6 + 2151780.61863184*cos(theta)**4 - 37861.242556572*cos(theta)**2 + 108.796674013138)*sin(2*phi)

@torch.jit.script
def Yl26_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.109617386791489*(1.0 - cos(theta)**2)**0.5*(192133811.96038*cos(theta)**25 - 1130198893.88459*cos(theta)**23 + 2917758368.90613*cos(theta)**21 - 4345597570.71126*cos(theta)**19 + 4128317692.17569*cos(theta)**17 - 2611400958.7716*cos(theta)**15 + 1114622360.45129*cos(theta)**13 - 318463531.557512*cos(theta)**11 + 59173967.0123756*cos(theta)**9 - 6762739.08712864*cos(theta)**7 + 430356.123726368*cos(theta)**5 - 12620.414185524*cos(theta)**3 + 108.796674013138*cos(theta))*sin(phi)

@torch.jit.script
def Yl26_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 15176214.4264154*cos(theta)**26 - 96711170.3644117*cos(theta)**24 + 272370234.903853*cos(theta)**22 - 446223576.331845*cos(theta)**20 + 471013775.016947*cos(theta)**18 - 335186546.872525*cos(theta)**16 + 163505632.620744*cos(theta)**14 - 54501877.540248*cos(theta)**12 + 12152445.667758*cos(theta)**10 - 1736063.66682257*cos(theta)**8 + 147302.3717304*cos(theta)**6 - 6479.57646907918*cos(theta)**4 + 111.716835673779*cos(theta)**2 - 0.318281583116179

@torch.jit.script
def Yl26_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.109617386791489*(1.0 - cos(theta)**2)**0.5*(192133811.96038*cos(theta)**25 - 1130198893.88459*cos(theta)**23 + 2917758368.90613*cos(theta)**21 - 4345597570.71126*cos(theta)**19 + 4128317692.17569*cos(theta)**17 - 2611400958.7716*cos(theta)**15 + 1114622360.45129*cos(theta)**13 - 318463531.557512*cos(theta)**11 + 59173967.0123756*cos(theta)**9 - 6762739.08712864*cos(theta)**7 + 430356.123726368*cos(theta)**5 - 12620.414185524*cos(theta)**3 + 108.796674013138*cos(theta))*cos(phi)

@torch.jit.script
def Yl26_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00414314778312938*(1.0 - cos(theta)**2)*(4803345299.0095*cos(theta)**24 - 25994574559.3455*cos(theta)**22 + 61272925747.0287*cos(theta)**20 - 82566353843.5138*cos(theta)**18 + 70181400766.9868*cos(theta)**16 - 39171014381.574*cos(theta)**14 + 14490090685.8668*cos(theta)**12 - 3503098847.13264*cos(theta)**10 + 532565703.11138*cos(theta)**8 - 47339173.6099005*cos(theta)**6 + 2151780.61863184*cos(theta)**4 - 37861.242556572*cos(theta)**2 + 108.796674013138)*cos(2*phi)

@torch.jit.script
def Yl26_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000157045611432433*(1.0 - cos(theta)**2)**1.5*(115280287176.228*cos(theta)**23 - 571880640305.601*cos(theta)**21 + 1225458514940.57*cos(theta)**19 - 1486194369183.25*cos(theta)**17 + 1122902412271.79*cos(theta)**15 - 548394201342.036*cos(theta)**13 + 173881088230.402*cos(theta)**11 - 35030988471.3264*cos(theta)**9 + 4260525624.89104*cos(theta)**7 - 284035041.659403*cos(theta)**5 + 8607122.47452736*cos(theta)**3 - 75722.4851131439*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl26_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.97862425042808e-6*(1.0 - cos(theta)**2)**2*(2651446605053.24*cos(theta)**22 - 12009493446417.6*cos(theta)**20 + 23283711783870.9*cos(theta)**18 - 25265304276115.2*cos(theta)**16 + 16843536184076.8*cos(theta)**14 - 7129124617446.47*cos(theta)**12 + 1912691970534.42*cos(theta)**10 - 315278896241.937*cos(theta)**8 + 29823679374.2373*cos(theta)**6 - 1420175208.29701*cos(theta)**4 + 25821367.4235821*cos(theta)**2 - 75722.4851131439)*cos(4*phi)

@torch.jit.script
def Yl26_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.28933354565387e-7*(1.0 - cos(theta)**2)**2.5*(58331825311171.3*cos(theta)**21 - 240189868928353.0*cos(theta)**19 + 419106812109676.0*cos(theta)**17 - 404244868417844.0*cos(theta)**15 + 235809506577076.0*cos(theta)**13 - 85549495409357.6*cos(theta)**11 + 19126919705344.2*cos(theta)**9 - 2522231169935.5*cos(theta)**7 + 178942076245.424*cos(theta)**5 - 5680700833.18806*cos(theta)**3 + 51642734.8471642*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl26_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.83129588187465e-9*(1.0 - cos(theta)**2)**3*(1.2249683315346e+15*cos(theta)**20 - 4.5636075096387e+15*cos(theta)**18 + 7.1248158058645e+15*cos(theta)**16 - 6.06367302626766e+15*cos(theta)**14 + 3.06552358550198e+15*cos(theta)**12 - 941044449502934.0*cos(theta)**10 + 172142277348098.0*cos(theta)**8 - 17655618189548.5*cos(theta)**6 + 894710381227.119*cos(theta)**4 - 17042102499.5642*cos(theta)**2 + 51642734.8471642)*cos(6*phi)

@torch.jit.script
def Yl26_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.43757725980871e-10*(1.0 - cos(theta)**2)**3.5*(2.4499366630692e+16*cos(theta)**19 - 8.21449351734965e+16*cos(theta)**17 + 1.13997052893832e+17*cos(theta)**15 - 8.48914223677472e+16*cos(theta)**13 + 3.67862830260238e+16*cos(theta)**11 - 9.41044449502934e+15*cos(theta)**9 + 1.37713821878478e+15*cos(theta)**7 - 105933709137291.0*cos(theta)**5 + 3578841524908.48*cos(theta)**3 - 34084204999.1283*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl26_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.35249668324814e-11*(1.0 - cos(theta)**2)**4*(4.65487965983147e+17*cos(theta)**18 - 1.39646389794944e+18*cos(theta)**16 + 1.70995579340748e+18*cos(theta)**14 - 1.10358849078071e+18*cos(theta)**12 + 4.04649113286262e+17*cos(theta)**10 - 8.46940004552641e+16*cos(theta)**8 + 9.63996753149347e+15*cos(theta)**6 - 529668545686454.0*cos(theta)**4 + 10736524574725.4*cos(theta)**2 - 34084204999.1283)*cos(8*phi)

@torch.jit.script
def Yl26_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.38847576616016e-13*(1.0 - cos(theta)**2)**4.5*(8.37878338769665e+18*cos(theta)**17 - 2.23434223671911e+19*cos(theta)**15 + 2.39393811077047e+19*cos(theta)**13 - 1.32430618893686e+19*cos(theta)**11 + 4.04649113286262e+18*cos(theta)**9 - 6.77552003642113e+17*cos(theta)**7 + 5.78398051889608e+16*cos(theta)**5 - 2.11867418274582e+15*cos(theta)**3 + 21473049149450.9*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl26_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.17816222989798e-14*(1.0 - cos(theta)**2)**5*(1.42439317590843e+20*cos(theta)**16 - 3.35151335507866e+20*cos(theta)**14 + 3.11211954400161e+20*cos(theta)**12 - 1.45673680783054e+20*cos(theta)**10 + 3.64184201957635e+19*cos(theta)**8 - 4.74286402549479e+18*cos(theta)**6 + 2.89199025944804e+17*cos(theta)**4 - 6.35602254823745e+15*cos(theta)**2 + 21473049149450.9)*cos(10*phi)

@torch.jit.script
def Yl26_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.95219161955017e-16*(1.0 - cos(theta)**2)**5.5*(2.27902908145349e+21*cos(theta)**15 - 4.69211869711012e+21*cos(theta)**13 + 3.73454345280193e+21*cos(theta)**11 - 1.45673680783054e+21*cos(theta)**9 + 2.91347361566108e+20*cos(theta)**7 - 2.84571841529687e+19*cos(theta)**5 + 1.15679610377922e+18*cos(theta)**3 - 1.27120450964749e+16*cos(theta))*cos(11*phi)

@torch.jit.script
def Yl26_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.74966044762475e-17*(1.0 - cos(theta)**2)**6*(3.41854362218023e+22*cos(theta)**14 - 6.09975430624316e+22*cos(theta)**12 + 4.10799779808213e+22*cos(theta)**10 - 1.31106312704749e+22*cos(theta)**8 + 2.03943153096276e+21*cos(theta)**6 - 1.42285920764844e+20*cos(theta)**4 + 3.47038831133765e+18*cos(theta)**2 - 1.27120450964749e+16)*cos(12*phi)

@torch.jit.script
def Yl26_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.60470653191418e-18*(1.0 - cos(theta)**2)**6.5*(4.78596107105233e+23*cos(theta)**13 - 7.31970516749179e+23*cos(theta)**11 + 4.10799779808213e+23*cos(theta)**9 - 1.04885050163799e+23*cos(theta)**7 + 1.22365891857766e+22*cos(theta)**5 - 5.69143683059374e+20*cos(theta)**3 + 6.9407766226753e+18*cos(theta))*cos(13*phi)

@torch.jit.script
def Yl26_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.03710366224851e-20*(1.0 - cos(theta)**2)**7*(6.22174939236802e+24*cos(theta)**12 - 8.05167568424097e+24*cos(theta)**10 + 3.69719801827392e+24*cos(theta)**8 - 7.34195351146593e+23*cos(theta)**6 + 6.11829459288828e+22*cos(theta)**4 - 1.70743104917812e+21*cos(theta)**2 + 6.9407766226753e+18)*cos(14*phi)

@torch.jit.script
def Yl26_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.17257134412823e-21*(1.0 - cos(theta)**2)**7.5*(7.46609927084163e+25*cos(theta)**11 - 8.05167568424097e+25*cos(theta)**9 + 2.95775841461913e+25*cos(theta)**7 - 4.40517210687956e+24*cos(theta)**5 + 2.44731783715531e+23*cos(theta)**3 - 3.41486209835625e+21*cos(theta))*cos(15*phi)

@torch.jit.script
def Yl26_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.47601377103699e-22*(1.0 - cos(theta)**2)**8*(8.21270919792579e+26*cos(theta)**10 - 7.24650811581687e+26*cos(theta)**8 + 2.07043089023339e+26*cos(theta)**6 - 2.20258605343978e+25*cos(theta)**4 + 7.34195351146593e+23*cos(theta)**2 - 3.41486209835625e+21)*cos(16*phi)

@torch.jit.script
def Yl26_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.11797046507269e-24*(1.0 - cos(theta)**2)**8.5*(8.21270919792579e+27*cos(theta)**9 - 5.7972064926535e+27*cos(theta)**7 + 1.24225853414004e+27*cos(theta)**5 - 8.81034421375912e+25*cos(theta)**3 + 1.46839070229319e+24*cos(theta))*cos(17*phi)

@torch.jit.script
def Yl26_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.57691474264813e-25*(1.0 - cos(theta)**2)**9*(7.39143827813321e+28*cos(theta)**8 - 4.05804454485745e+28*cos(theta)**6 + 6.21129267070018e+27*cos(theta)**4 - 2.64310326412774e+26*cos(theta)**2 + 1.46839070229319e+24)*cos(18*phi)

@torch.jit.script
def Yl26_m19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.88519959716718e-26*(1.0 - cos(theta)**2)**9.5*(5.91315062250657e+29*cos(theta)**7 - 2.43482672691447e+29*cos(theta)**5 + 2.48451706828007e+28*cos(theta)**3 - 5.28620652825547e+26*cos(theta))*cos(19*phi)

@torch.jit.script
def Yl26_m20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.0505806618571e-27*(1.0 - cos(theta)**2)**10*(4.1392054357546e+30*cos(theta)**6 - 1.21741336345723e+30*cos(theta)**4 + 7.45355120484021e+28*cos(theta)**2 - 5.28620652825547e+26)*cos(20*phi)

@torch.jit.script
def Yl26_m21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.25611679988175e-29*(1.0 - cos(theta)**2)**10.5*(2.48352326145276e+31*cos(theta)**5 - 4.86965345382894e+30*cos(theta)**3 + 1.49071024096804e+29*cos(theta))*cos(21*phi)

@torch.jit.script
def Yl26_m22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.03830602964508e-30*(1.0 - cos(theta)**2)**11*(1.24176163072638e+32*cos(theta)**4 - 1.46089603614868e+31*cos(theta)**2 + 1.49071024096804e+29)*cos(22*phi)

@torch.jit.script
def Yl26_m23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.88450430688934e-31*(1.0 - cos(theta)**2)**11.5*(4.96704652290552e+32*cos(theta)**3 - 2.92179207229736e+31*cos(theta))*cos(23*phi)

@torch.jit.script
def Yl26_m24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.35518790424645e-32*(1.0 - cos(theta)**2)**12*(1.49011395687166e+33*cos(theta)**2 - 2.92179207229736e+31)*cos(24*phi)

@torch.jit.script
def Yl26_m25(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.94984237067387*(1.0 - cos(theta)**2)**12.5*cos(25*phi)*cos(theta)

@torch.jit.script
def Yl26_m26(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.963769731686801*(1.0 - cos(theta)**2)**13*cos(26*phi)

@torch.jit.script
def Yl27_m_minus_27(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.97265258980333*(1.0 - cos(theta)**2)**13.5*sin(27*phi)

@torch.jit.script
def Yl27_m_minus_26(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.14750762604425*(1.0 - cos(theta)**2)**13*sin(26*phi)*cos(theta)

@torch.jit.script
def Yl27_m_minus_25(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.65888737989014e-34*(1.0 - cos(theta)**2)**12.5*(7.89760397141977e+34*cos(theta)**2 - 1.49011395687166e+33)*sin(25*phi)

@torch.jit.script
def Yl27_m_minus_24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.81894847243549e-33*(1.0 - cos(theta)**2)**12*(2.63253465713992e+34*cos(theta)**3 - 1.49011395687166e+33*cos(theta))*sin(24*phi)

@torch.jit.script
def Yl27_m_minus_23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.31112080905536e-32*(1.0 - cos(theta)**2)**11.5*(6.58133664284981e+33*cos(theta)**4 - 7.45056978435828e+32*cos(theta)**2 + 7.30448018074341e+30)*sin(23*phi)

@torch.jit.script
def Yl27_m_minus_22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.31410358327182e-30*(1.0 - cos(theta)**2)**11*(1.31626732856996e+33*cos(theta)**5 - 2.48352326145276e+32*cos(theta)**3 + 7.30448018074341e+30*cos(theta))*sin(22*phi)

@torch.jit.script
def Yl27_m_minus_21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.25321827372525e-29*(1.0 - cos(theta)**2)**10.5*(2.19377888094994e+32*cos(theta)**6 - 6.2088081536319e+31*cos(theta)**4 + 3.6522400903717e+30*cos(theta)**2 - 2.48451706828007e+28)*sin(21*phi)

@torch.jit.script
def Yl27_m_minus_20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.13021731864148e-28*(1.0 - cos(theta)**2)**10*(3.13396982992848e+31*cos(theta)**7 - 1.24176163072638e+31*cos(theta)**5 + 1.21741336345723e+30*cos(theta)**3 - 2.48451706828007e+28*cos(theta))*sin(20*phi)

@torch.jit.script
def Yl27_m_minus_19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.00878852093215e-27*(1.0 - cos(theta)**2)**9.5*(3.9174622874106e+30*cos(theta)**8 - 2.0696027178773e+30*cos(theta)**6 + 3.04353340864309e+29*cos(theta)**4 - 1.24225853414004e+28*cos(theta)**2 + 6.60775816031934e+25)*sin(19*phi)

@torch.jit.script
def Yl27_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.62954739542083e-25*(1.0 - cos(theta)**2)**9*(4.35273587490067e+29*cos(theta)**9 - 2.95657531125328e+29*cos(theta)**7 + 6.08706681728617e+28*cos(theta)**5 - 4.14086178046679e+27*cos(theta)**3 + 6.60775816031934e+25*cos(theta))*sin(18*phi)

@torch.jit.script
def Yl27_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.45679204070083e-24*(1.0 - cos(theta)**2)**8.5*(4.35273587490067e+28*cos(theta)**10 - 3.69571913906661e+28*cos(theta)**8 + 1.01451113621436e+28*cos(theta)**6 - 1.0352154451167e+27*cos(theta)**4 + 3.30387908015967e+25*cos(theta)**2 - 1.46839070229319e+23)*sin(17*phi)

@torch.jit.script
def Yl27_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.60494248954183e-23*(1.0 - cos(theta)**2)**8*(3.95703261354606e+27*cos(theta)**11 - 4.1063545989629e+27*cos(theta)**9 + 1.44930162316337e+27*cos(theta)**7 - 2.07043089023339e+26*cos(theta)**5 + 1.10129302671989e+25*cos(theta)**3 - 1.46839070229319e+23*cos(theta))*sin(16*phi)

@torch.jit.script
def Yl27_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.72751085492761e-21*(1.0 - cos(theta)**2)**7.5*(3.29752717795505e+26*cos(theta)**12 - 4.1063545989629e+26*cos(theta)**10 + 1.81162702895422e+26*cos(theta)**8 - 3.45071815038899e+25*cos(theta)**6 + 2.75323256679972e+24*cos(theta)**4 - 7.34195351146593e+22*cos(theta)**2 + 2.84571841529687e+20)*sin(15*phi)

@torch.jit.script
def Yl27_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.03661292375851e-20*(1.0 - cos(theta)**2)**7*(2.53655936765773e+25*cos(theta)**13 - 3.73304963542081e+25*cos(theta)**11 + 2.01291892106024e+25*cos(theta)**9 - 4.92959735769855e+24*cos(theta)**7 + 5.50646513359945e+23*cos(theta)**5 - 2.44731783715531e+22*cos(theta)**3 + 2.84571841529687e+20*cos(theta))*sin(14*phi)

@torch.jit.script
def Yl27_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.67103717108456e-19*(1.0 - cos(theta)**2)**6.5*(1.81182811975552e+24*cos(theta)**14 - 3.11087469618401e+24*cos(theta)**12 + 2.01291892106024e+24*cos(theta)**10 - 6.16199669712319e+23*cos(theta)**8 + 9.17744188933241e+22*cos(theta)**6 - 6.11829459288828e+21*cos(theta)**4 + 1.42285920764844e+20*cos(theta)**2 - 4.95769758762521e+17)*sin(13*phi)

@torch.jit.script
def Yl27_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.36891063526465e-17*(1.0 - cos(theta)**2)**6*(1.20788541317035e+23*cos(theta)**15 - 2.39298053552616e+23*cos(theta)**13 + 1.82992629187295e+23*cos(theta)**11 - 6.84666299680355e+22*cos(theta)**9 + 1.31106312704749e+22*cos(theta)**7 - 1.22365891857766e+21*cos(theta)**5 + 4.74286402549479e+19*cos(theta)**3 - 4.95769758762521e+17*cos(theta))*sin(12*phi)

@torch.jit.script
def Yl27_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.91753687024496e-16*(1.0 - cos(theta)**2)**5.5*(7.54928383231468e+21*cos(theta)**16 - 1.70927181109012e+22*cos(theta)**14 + 1.52493857656079e+22*cos(theta)**12 - 6.84666299680355e+21*cos(theta)**10 + 1.63882890880936e+21*cos(theta)**8 - 2.03943153096276e+20*cos(theta)**6 + 1.1857160063737e+19*cos(theta)**4 - 2.47884879381261e+17*cos(theta)**2 + 794502818529682.0)*sin(11*phi)

@torch.jit.script
def Yl27_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.50403253709877e-14*(1.0 - cos(theta)**2)**5*(4.44075519547922e+20*cos(theta)**17 - 1.13951454072674e+21*cos(theta)**15 + 1.17302967427753e+21*cos(theta)**13 - 6.22423908800322e+20*cos(theta)**11 + 1.82092100978818e+20*cos(theta)**9 - 2.91347361566108e+19*cos(theta)**7 + 2.37143201274739e+18*cos(theta)**5 - 8.26282931270869e+16*cos(theta)**3 + 794502818529682.0*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl27_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.8814531289017e-13*(1.0 - cos(theta)**2)**4.5*(2.46708621971068e+19*cos(theta)**18 - 7.12196587954215e+19*cos(theta)**16 + 8.37878338769665e+19*cos(theta)**14 - 5.18686590666935e+19*cos(theta)**12 + 1.82092100978818e+19*cos(theta)**10 - 3.64184201957635e+18*cos(theta)**8 + 3.95238668791232e+17*cos(theta)**6 - 2.06570732817717e+16*cos(theta)**4 + 397251409264841.0*cos(theta)**2 - 1192947174969.49)*sin(9*phi)

@torch.jit.script
def Yl27_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.01513171657834e-11*(1.0 - cos(theta)**2)**4*(1.29846643142667e+18*cos(theta)**19 - 4.18939169384832e+18*cos(theta)**17 + 5.58585559179777e+18*cos(theta)**15 - 3.98989685128412e+18*cos(theta)**13 + 1.65538273617107e+18*cos(theta)**11 - 4.04649113286262e+17*cos(theta)**9 + 5.6462666970176e+16*cos(theta)**7 - 4.13141465635434e+15*cos(theta)**5 + 132417136421614.0*cos(theta)**3 - 1192947174969.49*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl27_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.68578607004038e-10*(1.0 - cos(theta)**2)**3.5*(6.49233215713337e+16*cos(theta)**20 - 2.32743982991574e+17*cos(theta)**18 + 3.4911597448736e+17*cos(theta)**16 - 2.8499263223458e+17*cos(theta)**14 + 1.37948561347589e+17*cos(theta)**12 - 4.04649113286262e+16*cos(theta)**10 + 7.05783337127201e+15*cos(theta)**8 - 688569109392391.0*cos(theta)**6 + 33104284105403.4*cos(theta)**4 - 596473587484.746*cos(theta)**2 + 1704210249.95642)*sin(7*phi)

@torch.jit.script
def Yl27_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.17662944926961e-9*(1.0 - cos(theta)**2)**3*(3.09158674149208e+15*cos(theta)**21 - 1.2249683315346e+16*cos(theta)**19 + 2.05362337933741e+16*cos(theta)**17 - 1.89995088156387e+16*cos(theta)**15 + 1.06114277959684e+16*cos(theta)**13 - 3.67862830260238e+15*cos(theta)**11 + 784203707919112.0*cos(theta)**9 - 98367015627484.4*cos(theta)**7 + 6620856821080.68*cos(theta)**5 - 198824529161.582*cos(theta)**3 + 1704210249.95642*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl27_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.93369882461158e-7*(1.0 - cos(theta)**2)**2.5*(140526670067822.0*cos(theta)**22 - 612484165767299.0*cos(theta)**20 + 1.14090187740967e+15*cos(theta)**18 - 1.18746930097742e+15*cos(theta)**16 + 757959128283457.0*cos(theta)**14 - 306552358550198.0*cos(theta)**12 + 78420370791911.2*cos(theta)**10 - 12295876953435.5*cos(theta)**8 + 1103476136846.78*cos(theta)**6 - 49706132290.3955*cos(theta)**4 + 852105124.978209*cos(theta)**2 - 2347397.03850746)*sin(5*phi)

@torch.jit.script
def Yl27_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.24599340659887e-6*(1.0 - cos(theta)**2)**2*(6109855220340.08*cos(theta)**23 - 29165912655585.7*cos(theta)**21 + 60047467232088.1*cos(theta)**19 - 69851135351612.7*cos(theta)**17 + 50530608552230.5*cos(theta)**15 - 23580950657707.6*cos(theta)**13 + 7129124617446.47*cos(theta)**11 - 1366208550381.73*cos(theta)**9 + 157639448120.969*cos(theta)**7 - 9941226458.0791*cos(theta)**5 + 284035041.659403*cos(theta)**3 - 2347397.03850746*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl27_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00014309162252077*(1.0 - cos(theta)**2)**1.5*(254577300847.503*cos(theta)**24 - 1325723302526.62*cos(theta)**22 + 3002373361604.41*cos(theta)**20 - 3880618630645.15*cos(theta)**18 + 3158163034514.4*cos(theta)**16 - 1684353618407.68*cos(theta)**14 + 594093718120.539*cos(theta)**12 - 136620855038.173*cos(theta)**10 + 19704931015.1211*cos(theta)**8 - 1656871076.34652*cos(theta)**6 + 71008760.4148507*cos(theta)**4 - 1173698.51925373*cos(theta)**2 + 3155.103546381)*sin(3*phi)

@torch.jit.script
def Yl27_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00391872547223201*(1.0 - cos(theta)**2)*(10183092033.9001*cos(theta)**25 - 57640143588.114*cos(theta)**23 + 142970160076.4*cos(theta)**21 - 204243085823.429*cos(theta)**19 + 185774296147.906*cos(theta)**17 - 112290241227.179*cos(theta)**15 + 45699516778.503*cos(theta)**13 - 12420077730.743*cos(theta)**11 + 2189436779.4579*cos(theta)**9 - 236695868.049502*cos(theta)**7 + 14201752.0829701*cos(theta)**5 - 391232.839751244*cos(theta)**3 + 3155.103546381*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl27_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.107604519572121*(1.0 - cos(theta)**2)**0.5*(391657385.919236*cos(theta)**26 - 2401672649.50475*cos(theta)**24 + 6498643639.83638*cos(theta)**22 - 10212154291.1714*cos(theta)**20 + 10320794230.4392*cos(theta)**18 - 7018140076.69868*cos(theta)**16 + 3264251198.4645*cos(theta)**14 - 1035006477.56191*cos(theta)**12 + 218943677.94579*cos(theta)**10 - 29586983.5061878*cos(theta)**8 + 2366958.68049502*cos(theta)**6 - 97808.2099378109*cos(theta)**4 + 1577.5517731905*cos(theta)**2 - 4.18448746204376)*sin(phi)

@torch.jit.script
def Yl27_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 30347223.9434462*cos(theta)**27 - 200978784.983955*cos(theta)**25 + 591114073.482221*cos(theta)**23 - 1017359595.85716*cos(theta)**21 + 1136412314.52129*cos(theta)**19 - 863673359.036181*cos(theta)**17 + 455269677.631475*cos(theta)**15 - 166562077.182247*cos(theta)**13 + 41640519.2955618*cos(theta)**11 - 6877563.24701471*cos(theta)**9 + 707406.505407227*cos(theta)**7 - 40924.3432880214*cos(theta)**5 + 1100.11675505434*cos(theta)**3 - 8.75424473518041*cos(theta)

@torch.jit.script
def Yl27_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.107604519572121*(1.0 - cos(theta)**2)**0.5*(391657385.919236*cos(theta)**26 - 2401672649.50475*cos(theta)**24 + 6498643639.83638*cos(theta)**22 - 10212154291.1714*cos(theta)**20 + 10320794230.4392*cos(theta)**18 - 7018140076.69868*cos(theta)**16 + 3264251198.4645*cos(theta)**14 - 1035006477.56191*cos(theta)**12 + 218943677.94579*cos(theta)**10 - 29586983.5061878*cos(theta)**8 + 2366958.68049502*cos(theta)**6 - 97808.2099378109*cos(theta)**4 + 1577.5517731905*cos(theta)**2 - 4.18448746204376)*cos(phi)

@torch.jit.script
def Yl27_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00391872547223201*(1.0 - cos(theta)**2)*(10183092033.9001*cos(theta)**25 - 57640143588.114*cos(theta)**23 + 142970160076.4*cos(theta)**21 - 204243085823.429*cos(theta)**19 + 185774296147.906*cos(theta)**17 - 112290241227.179*cos(theta)**15 + 45699516778.503*cos(theta)**13 - 12420077730.743*cos(theta)**11 + 2189436779.4579*cos(theta)**9 - 236695868.049502*cos(theta)**7 + 14201752.0829701*cos(theta)**5 - 391232.839751244*cos(theta)**3 + 3155.103546381*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl27_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00014309162252077*(1.0 - cos(theta)**2)**1.5*(254577300847.503*cos(theta)**24 - 1325723302526.62*cos(theta)**22 + 3002373361604.41*cos(theta)**20 - 3880618630645.15*cos(theta)**18 + 3158163034514.4*cos(theta)**16 - 1684353618407.68*cos(theta)**14 + 594093718120.539*cos(theta)**12 - 136620855038.173*cos(theta)**10 + 19704931015.1211*cos(theta)**8 - 1656871076.34652*cos(theta)**6 + 71008760.4148507*cos(theta)**4 - 1173698.51925373*cos(theta)**2 + 3155.103546381)*cos(3*phi)

@torch.jit.script
def Yl27_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.24599340659887e-6*(1.0 - cos(theta)**2)**2*(6109855220340.08*cos(theta)**23 - 29165912655585.7*cos(theta)**21 + 60047467232088.1*cos(theta)**19 - 69851135351612.7*cos(theta)**17 + 50530608552230.5*cos(theta)**15 - 23580950657707.6*cos(theta)**13 + 7129124617446.47*cos(theta)**11 - 1366208550381.73*cos(theta)**9 + 157639448120.969*cos(theta)**7 - 9941226458.0791*cos(theta)**5 + 284035041.659403*cos(theta)**3 - 2347397.03850746*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl27_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.93369882461158e-7*(1.0 - cos(theta)**2)**2.5*(140526670067822.0*cos(theta)**22 - 612484165767299.0*cos(theta)**20 + 1.14090187740967e+15*cos(theta)**18 - 1.18746930097742e+15*cos(theta)**16 + 757959128283457.0*cos(theta)**14 - 306552358550198.0*cos(theta)**12 + 78420370791911.2*cos(theta)**10 - 12295876953435.5*cos(theta)**8 + 1103476136846.78*cos(theta)**6 - 49706132290.3955*cos(theta)**4 + 852105124.978209*cos(theta)**2 - 2347397.03850746)*cos(5*phi)

@torch.jit.script
def Yl27_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.17662944926961e-9*(1.0 - cos(theta)**2)**3*(3.09158674149208e+15*cos(theta)**21 - 1.2249683315346e+16*cos(theta)**19 + 2.05362337933741e+16*cos(theta)**17 - 1.89995088156387e+16*cos(theta)**15 + 1.06114277959684e+16*cos(theta)**13 - 3.67862830260238e+15*cos(theta)**11 + 784203707919112.0*cos(theta)**9 - 98367015627484.4*cos(theta)**7 + 6620856821080.68*cos(theta)**5 - 198824529161.582*cos(theta)**3 + 1704210249.95642*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl27_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.68578607004038e-10*(1.0 - cos(theta)**2)**3.5*(6.49233215713337e+16*cos(theta)**20 - 2.32743982991574e+17*cos(theta)**18 + 3.4911597448736e+17*cos(theta)**16 - 2.8499263223458e+17*cos(theta)**14 + 1.37948561347589e+17*cos(theta)**12 - 4.04649113286262e+16*cos(theta)**10 + 7.05783337127201e+15*cos(theta)**8 - 688569109392391.0*cos(theta)**6 + 33104284105403.4*cos(theta)**4 - 596473587484.746*cos(theta)**2 + 1704210249.95642)*cos(7*phi)

@torch.jit.script
def Yl27_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.01513171657834e-11*(1.0 - cos(theta)**2)**4*(1.29846643142667e+18*cos(theta)**19 - 4.18939169384832e+18*cos(theta)**17 + 5.58585559179777e+18*cos(theta)**15 - 3.98989685128412e+18*cos(theta)**13 + 1.65538273617107e+18*cos(theta)**11 - 4.04649113286262e+17*cos(theta)**9 + 5.6462666970176e+16*cos(theta)**7 - 4.13141465635434e+15*cos(theta)**5 + 132417136421614.0*cos(theta)**3 - 1192947174969.49*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl27_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.8814531289017e-13*(1.0 - cos(theta)**2)**4.5*(2.46708621971068e+19*cos(theta)**18 - 7.12196587954215e+19*cos(theta)**16 + 8.37878338769665e+19*cos(theta)**14 - 5.18686590666935e+19*cos(theta)**12 + 1.82092100978818e+19*cos(theta)**10 - 3.64184201957635e+18*cos(theta)**8 + 3.95238668791232e+17*cos(theta)**6 - 2.06570732817717e+16*cos(theta)**4 + 397251409264841.0*cos(theta)**2 - 1192947174969.49)*cos(9*phi)

@torch.jit.script
def Yl27_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.50403253709877e-14*(1.0 - cos(theta)**2)**5*(4.44075519547922e+20*cos(theta)**17 - 1.13951454072674e+21*cos(theta)**15 + 1.17302967427753e+21*cos(theta)**13 - 6.22423908800322e+20*cos(theta)**11 + 1.82092100978818e+20*cos(theta)**9 - 2.91347361566108e+19*cos(theta)**7 + 2.37143201274739e+18*cos(theta)**5 - 8.26282931270869e+16*cos(theta)**3 + 794502818529682.0*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl27_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.91753687024496e-16*(1.0 - cos(theta)**2)**5.5*(7.54928383231468e+21*cos(theta)**16 - 1.70927181109012e+22*cos(theta)**14 + 1.52493857656079e+22*cos(theta)**12 - 6.84666299680355e+21*cos(theta)**10 + 1.63882890880936e+21*cos(theta)**8 - 2.03943153096276e+20*cos(theta)**6 + 1.1857160063737e+19*cos(theta)**4 - 2.47884879381261e+17*cos(theta)**2 + 794502818529682.0)*cos(11*phi)

@torch.jit.script
def Yl27_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.36891063526465e-17*(1.0 - cos(theta)**2)**6*(1.20788541317035e+23*cos(theta)**15 - 2.39298053552616e+23*cos(theta)**13 + 1.82992629187295e+23*cos(theta)**11 - 6.84666299680355e+22*cos(theta)**9 + 1.31106312704749e+22*cos(theta)**7 - 1.22365891857766e+21*cos(theta)**5 + 4.74286402549479e+19*cos(theta)**3 - 4.95769758762521e+17*cos(theta))*cos(12*phi)

@torch.jit.script
def Yl27_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.67103717108456e-19*(1.0 - cos(theta)**2)**6.5*(1.81182811975552e+24*cos(theta)**14 - 3.11087469618401e+24*cos(theta)**12 + 2.01291892106024e+24*cos(theta)**10 - 6.16199669712319e+23*cos(theta)**8 + 9.17744188933241e+22*cos(theta)**6 - 6.11829459288828e+21*cos(theta)**4 + 1.42285920764844e+20*cos(theta)**2 - 4.95769758762521e+17)*cos(13*phi)

@torch.jit.script
def Yl27_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.03661292375851e-20*(1.0 - cos(theta)**2)**7*(2.53655936765773e+25*cos(theta)**13 - 3.73304963542081e+25*cos(theta)**11 + 2.01291892106024e+25*cos(theta)**9 - 4.92959735769855e+24*cos(theta)**7 + 5.50646513359945e+23*cos(theta)**5 - 2.44731783715531e+22*cos(theta)**3 + 2.84571841529687e+20*cos(theta))*cos(14*phi)

@torch.jit.script
def Yl27_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.72751085492761e-21*(1.0 - cos(theta)**2)**7.5*(3.29752717795505e+26*cos(theta)**12 - 4.1063545989629e+26*cos(theta)**10 + 1.81162702895422e+26*cos(theta)**8 - 3.45071815038899e+25*cos(theta)**6 + 2.75323256679972e+24*cos(theta)**4 - 7.34195351146593e+22*cos(theta)**2 + 2.84571841529687e+20)*cos(15*phi)

@torch.jit.script
def Yl27_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.60494248954183e-23*(1.0 - cos(theta)**2)**8*(3.95703261354606e+27*cos(theta)**11 - 4.1063545989629e+27*cos(theta)**9 + 1.44930162316337e+27*cos(theta)**7 - 2.07043089023339e+26*cos(theta)**5 + 1.10129302671989e+25*cos(theta)**3 - 1.46839070229319e+23*cos(theta))*cos(16*phi)

@torch.jit.script
def Yl27_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.45679204070083e-24*(1.0 - cos(theta)**2)**8.5*(4.35273587490067e+28*cos(theta)**10 - 3.69571913906661e+28*cos(theta)**8 + 1.01451113621436e+28*cos(theta)**6 - 1.0352154451167e+27*cos(theta)**4 + 3.30387908015967e+25*cos(theta)**2 - 1.46839070229319e+23)*cos(17*phi)

@torch.jit.script
def Yl27_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.62954739542083e-25*(1.0 - cos(theta)**2)**9*(4.35273587490067e+29*cos(theta)**9 - 2.95657531125328e+29*cos(theta)**7 + 6.08706681728617e+28*cos(theta)**5 - 4.14086178046679e+27*cos(theta)**3 + 6.60775816031934e+25*cos(theta))*cos(18*phi)

@torch.jit.script
def Yl27_m19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.00878852093215e-27*(1.0 - cos(theta)**2)**9.5*(3.9174622874106e+30*cos(theta)**8 - 2.0696027178773e+30*cos(theta)**6 + 3.04353340864309e+29*cos(theta)**4 - 1.24225853414004e+28*cos(theta)**2 + 6.60775816031934e+25)*cos(19*phi)

@torch.jit.script
def Yl27_m20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.13021731864148e-28*(1.0 - cos(theta)**2)**10*(3.13396982992848e+31*cos(theta)**7 - 1.24176163072638e+31*cos(theta)**5 + 1.21741336345723e+30*cos(theta)**3 - 2.48451706828007e+28*cos(theta))*cos(20*phi)

@torch.jit.script
def Yl27_m21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.25321827372525e-29*(1.0 - cos(theta)**2)**10.5*(2.19377888094994e+32*cos(theta)**6 - 6.2088081536319e+31*cos(theta)**4 + 3.6522400903717e+30*cos(theta)**2 - 2.48451706828007e+28)*cos(21*phi)

@torch.jit.script
def Yl27_m22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.31410358327182e-30*(1.0 - cos(theta)**2)**11*(1.31626732856996e+33*cos(theta)**5 - 2.48352326145276e+32*cos(theta)**3 + 7.30448018074341e+30*cos(theta))*cos(22*phi)

@torch.jit.script
def Yl27_m23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.31112080905536e-32*(1.0 - cos(theta)**2)**11.5*(6.58133664284981e+33*cos(theta)**4 - 7.45056978435828e+32*cos(theta)**2 + 7.30448018074341e+30)*cos(23*phi)

@torch.jit.script
def Yl27_m24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.81894847243549e-33*(1.0 - cos(theta)**2)**12*(2.63253465713992e+34*cos(theta)**3 - 1.49011395687166e+33*cos(theta))*cos(24*phi)

@torch.jit.script
def Yl27_m25(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.65888737989014e-34*(1.0 - cos(theta)**2)**12.5*(7.89760397141977e+34*cos(theta)**2 - 1.49011395687166e+33)*cos(25*phi)

@torch.jit.script
def Yl27_m26(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.14750762604425*(1.0 - cos(theta)**2)**13*cos(26*phi)*cos(theta)

@torch.jit.script
def Yl27_m27(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.97265258980333*(1.0 - cos(theta)**2)**13.5*cos(27*phi)

@torch.jit.script
def Yl28_m_minus_28(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.981298560633835*(1.0 - cos(theta)**2)**14*sin(28*phi)

@torch.jit.script
def Yl28_m_minus_27(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.34336601605245*(1.0 - cos(theta)**2)**13.5*sin(27*phi)*cos(theta)

@torch.jit.script
def Yl28_m_minus_26(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.86550503264189e-36*(1.0 - cos(theta)**2)**13*(4.34368218428088e+36*cos(theta)**2 - 7.89760397141977e+34)*sin(26*phi)

@torch.jit.script
def Yl28_m_minus_25(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.12839457090042e-34*(1.0 - cos(theta)**2)**12.5*(1.44789406142696e+36*cos(theta)**3 - 7.89760397141977e+34*cos(theta))*sin(25*phi)

@torch.jit.script
def Yl28_m_minus_24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.64296729492452e-33*(1.0 - cos(theta)**2)**12*(3.6197351535674e+35*cos(theta)**4 - 3.94880198570989e+34*cos(theta)**2 + 3.72528489217914e+32)*sin(24*phi)

@torch.jit.script
def Yl28_m_minus_23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.64920516074126e-32*(1.0 - cos(theta)**2)**11.5*(7.23947030713479e+34*cos(theta)**5 - 1.31626732856996e+34*cos(theta)**3 + 3.72528489217914e+32*cos(theta))*sin(23*phi)

@torch.jit.script
def Yl28_m_minus_22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.63421635555746e-31*(1.0 - cos(theta)**2)**11*(1.20657838452247e+34*cos(theta)**6 - 3.29066832142491e+33*cos(theta)**4 + 1.86264244608957e+32*cos(theta)**2 - 1.21741336345723e+30)*sin(22*phi)

@torch.jit.script
def Yl28_m_minus_21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.66982492934009e-30*(1.0 - cos(theta)**2)**10.5*(1.72368340646067e+33*cos(theta)**7 - 6.58133664284981e+32*cos(theta)**5 + 6.2088081536319e+31*cos(theta)**3 - 1.21741336345723e+30*cos(theta))*sin(21*phi)

@torch.jit.script
def Yl28_m_minus_20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.71653775978624e-28*(1.0 - cos(theta)**2)**10*(2.15460425807583e+32*cos(theta)**8 - 1.09688944047497e+32*cos(theta)**6 + 1.55220203840797e+31*cos(theta)**4 - 6.08706681728617e+29*cos(theta)**2 + 3.10564633535009e+27)*sin(20*phi)

@torch.jit.script
def Yl28_m_minus_19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.56775673567227e-27*(1.0 - cos(theta)**2)**9.5*(2.39400473119537e+31*cos(theta)**9 - 1.56698491496424e+31*cos(theta)**7 + 3.10440407681595e+30*cos(theta)**5 - 2.02902227242872e+29*cos(theta)**3 + 3.10564633535009e+27*cos(theta))*sin(19*phi)

@torch.jit.script
def Yl28_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.73471228858538e-26*(1.0 - cos(theta)**2)**9*(2.39400473119537e+30*cos(theta)**10 - 1.9587311437053e+30*cos(theta)**8 + 5.17400679469325e+29*cos(theta)**6 - 5.07255568107181e+28*cos(theta)**4 + 1.55282316767504e+27*cos(theta)**2 - 6.60775816031934e+24)*sin(18*phi)

@torch.jit.script
def Yl28_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.7398805056302e-24*(1.0 - cos(theta)**2)**8.5*(2.17636793745033e+29*cos(theta)**11 - 2.17636793745033e+29*cos(theta)**9 + 7.39143827813321e+28*cos(theta)**7 - 1.01451113621436e+28*cos(theta)**5 + 5.17607722558348e+26*cos(theta)**3 - 6.60775816031934e+24*cos(theta))*sin(17*phi)

@torch.jit.script
def Yl28_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.04311693361802e-23*(1.0 - cos(theta)**2)**8*(1.81363994787528e+28*cos(theta)**12 - 2.17636793745033e+28*cos(theta)**10 + 9.23929784766651e+27*cos(theta)**8 - 1.6908518936906e+27*cos(theta)**6 + 1.29401930639587e+26*cos(theta)**4 - 3.30387908015967e+24*cos(theta)**2 + 1.22365891857766e+22)*sin(16*phi)

@torch.jit.script
def Yl28_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.66972930141058e-22*(1.0 - cos(theta)**2)**7.5*(1.39510765221175e+27*cos(theta)**13 - 1.97851630677303e+27*cos(theta)**11 + 1.02658864974072e+27*cos(theta)**9 - 2.41550270527229e+26*cos(theta)**7 + 2.58803861279174e+25*cos(theta)**5 - 1.10129302671989e+24*cos(theta)**3 + 1.22365891857766e+22*cos(theta))*sin(15*phi)

@torch.jit.script
def Yl28_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.3725346401488e-20*(1.0 - cos(theta)**2)**7*(9.96505465865538e+25*cos(theta)**14 - 1.64876358897753e+26*cos(theta)**12 + 1.02658864974072e+26*cos(theta)**10 - 3.01937838159036e+25*cos(theta)**8 + 4.31339768798623e+24*cos(theta)**6 - 2.75323256679972e+23*cos(theta)**4 + 6.11829459288828e+21*cos(theta)**2 - 2.03265601092634e+19)*sin(14*phi)

@torch.jit.script
def Yl28_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.95501468493973e-19*(1.0 - cos(theta)**2)**6.5*(6.64336977243692e+24*cos(theta)**15 - 1.26827968382887e+25*cos(theta)**13 + 9.33262408855204e+24*cos(theta)**11 - 3.35486486843374e+24*cos(theta)**9 + 6.16199669712319e+23*cos(theta)**7 - 5.50646513359945e+22*cos(theta)**5 + 2.03943153096276e+21*cos(theta)**3 - 2.03265601092634e+19*cos(theta))*sin(13*phi)

@torch.jit.script
def Yl28_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.52522795453625e-17*(1.0 - cos(theta)**2)**6*(4.15210610777307e+23*cos(theta)**16 - 9.05914059877762e+23*cos(theta)**14 + 7.77718674046003e+23*cos(theta)**12 - 3.35486486843374e+23*cos(theta)**10 + 7.70249587140399e+22*cos(theta)**8 - 9.17744188933241e+21*cos(theta)**6 + 5.0985788274069e+20*cos(theta)**4 - 1.01632800546317e+19*cos(theta)**2 + 3.09856099226576e+16)*sin(12*phi)

@torch.jit.script
def Yl28_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.977307899878e-16*(1.0 - cos(theta)**2)**5.5*(2.44241535751357e+22*cos(theta)**17 - 6.03942706585174e+22*cos(theta)**15 + 5.98245133881541e+22*cos(theta)**13 - 3.04987715312158e+22*cos(theta)**11 + 8.55832874600443e+21*cos(theta)**9 - 1.31106312704749e+21*cos(theta)**7 + 1.01971576548138e+20*cos(theta)**5 - 3.38776001821056e+18*cos(theta)**3 + 3.09856099226576e+16*cos(theta))*sin(11*phi)

@torch.jit.script
def Yl28_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.05379896790437e-14*(1.0 - cos(theta)**2)**5*(1.35689742084087e+21*cos(theta)**18 - 3.77464191615734e+21*cos(theta)**16 + 4.27317952772529e+21*cos(theta)**14 - 2.54156429426798e+21*cos(theta)**12 + 8.55832874600443e+20*cos(theta)**10 - 1.63882890880936e+20*cos(theta)**8 + 1.6995262758023e+19*cos(theta)**6 - 8.46940004552641e+17*cos(theta)**4 + 1.54928049613288e+16*cos(theta)**2 - 44139045473871.2)*sin(10*phi)

@torch.jit.script
def Yl28_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.83156390560776e-13*(1.0 - cos(theta)**2)**4.5*(7.1415653728467e+19*cos(theta)**19 - 2.22037759773961e+20*cos(theta)**17 + 2.84878635181686e+20*cos(theta)**15 - 1.95504945712922e+20*cos(theta)**13 + 7.78029886000403e+19*cos(theta)**11 - 1.82092100978818e+19*cos(theta)**9 + 2.42789467971757e+18*cos(theta)**7 - 1.69388000910528e+17*cos(theta)**5 + 5.16426832044293e+15*cos(theta)**3 - 44139045473871.2*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl28_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.70268659114473e-12*(1.0 - cos(theta)**2)**4*(3.57078268642335e+18*cos(theta)**20 - 1.23354310985534e+19*cos(theta)**18 + 1.78049146988554e+19*cos(theta)**16 - 1.39646389794944e+19*cos(theta)**14 + 6.48358238333669e+18*cos(theta)**12 - 1.82092100978818e+18*cos(theta)**10 + 3.03486834964696e+17*cos(theta)**8 - 2.8231333485088e+16*cos(theta)**6 + 1.29106708011073e+15*cos(theta)**4 - 22069522736935.6*cos(theta)**2 + 59647358748.4746)*sin(8*phi)

@torch.jit.script
def Yl28_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.11788866150653e-10*(1.0 - cos(theta)**2)**3.5*(1.70037270782064e+17*cos(theta)**21 - 6.49233215713337e+17*cos(theta)**19 + 1.04734792346208e+18*cos(theta)**17 - 9.30975931966294e+17*cos(theta)**15 + 4.98737106410515e+17*cos(theta)**13 - 1.65538273617107e+17*cos(theta)**11 + 3.37207594405218e+16*cos(theta)**9 - 4.03304764072686e+15*cos(theta)**7 + 258213416022147.0*cos(theta)**5 - 7356507578978.53*cos(theta)**3 + 59647358748.4746*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl28_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.8769025298657e-9*(1.0 - cos(theta)**2)**3*(7.7289668537302e+15*cos(theta)**22 - 3.24616607856668e+16*cos(theta)**20 + 5.81859957478934e+16*cos(theta)**18 - 5.81859957478934e+16*cos(theta)**16 + 3.56240790293225e+16*cos(theta)**14 - 1.37948561347589e+16*cos(theta)**12 + 3.37207594405218e+15*cos(theta)**10 - 504130955090858.0*cos(theta)**8 + 43035569337024.4*cos(theta)**6 - 1839126894744.63*cos(theta)**4 + 29823679374.2373*cos(theta)**2 - 77464102.2707462)*sin(6*phi)

@torch.jit.script
def Yl28_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.64343247431143e-7*(1.0 - cos(theta)**2)**2.5*(336042037118704.0*cos(theta)**23 - 1.54579337074604e+15*cos(theta)**21 + 3.06242082883649e+15*cos(theta)**19 - 3.42270563222902e+15*cos(theta)**17 + 2.37493860195483e+15*cos(theta)**15 - 1.06114277959684e+15*cos(theta)**13 + 306552358550198.0*cos(theta)**11 - 56014550565650.8*cos(theta)**9 + 6147938476717.77*cos(theta)**7 - 367825378948.927*cos(theta)**5 + 9941226458.0791*cos(theta)**3 - 77464102.2707462*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl28_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.62502894662956e-6*(1.0 - cos(theta)**2)**2*(14001751546612.7*cos(theta)**24 - 70263335033910.9*cos(theta)**22 + 153121041441825.0*cos(theta)**20 - 190150312901612.0*cos(theta)**18 + 148433662622177.0*cos(theta)**16 - 75795912828345.7*cos(theta)**14 + 25546029879183.2*cos(theta)**12 - 5601455056565.08*cos(theta)**10 + 768492309589.722*cos(theta)**8 - 61304229824.8211*cos(theta)**6 + 2485306614.51977*cos(theta)**4 - 38732051.1353731*cos(theta)**2 + 97808.2099378109)*sin(4*phi)

@torch.jit.script
def Yl28_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000130815573253833*(1.0 - cos(theta)**2)**1.5*(560070061864.507*cos(theta)**25 - 3054927610170.04*cos(theta)**23 + 7291478163896.42*cos(theta)**21 - 10007911205348.0*cos(theta)**19 + 8731391918951.59*cos(theta)**17 - 5053060855223.05*cos(theta)**15 + 1965079221475.63*cos(theta)**13 - 509223186960.462*cos(theta)**11 + 85388034398.858*cos(theta)**9 - 8757747117.83159*cos(theta)**7 + 497061322.903955*cos(theta)**5 - 12910683.711791*cos(theta)**3 + 97808.2099378109*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl28_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00371387232545999*(1.0 - cos(theta)**2)*(21541156225.558*cos(theta)**26 - 127288650423.752*cos(theta)**24 + 331430825631.655*cos(theta)**22 - 500395560267.401*cos(theta)**20 + 485077328830.644*cos(theta)**18 - 315816303451.44*cos(theta)**16 + 140362801533.974*cos(theta)**14 - 42435265580.0385*cos(theta)**12 + 8538803439.8858*cos(theta)**10 - 1094718389.72895*cos(theta)**8 + 82843553.8173258*cos(theta)**6 - 3227670.92794776*cos(theta)**4 + 48904.1049689054*cos(theta)**2 - 121.350136399269)*sin(2*phi)

@torch.jit.script
def Yl28_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.105698659387677*(1.0 - cos(theta)**2)**0.5*(797820600.946591*cos(theta)**27 - 5091546016.95007*cos(theta)**25 + 14410035897.0285*cos(theta)**23 - 23828360012.7334*cos(theta)**21 + 25530385727.9286*cos(theta)**19 - 18577429614.7906*cos(theta)**17 + 9357520102.2649*cos(theta)**15 - 3264251198.4645*cos(theta)**13 + 776254858.171436*cos(theta)**11 - 121635376.63655*cos(theta)**9 + 11834793.4024751*cos(theta)**7 - 645534.185589552*cos(theta)**5 + 16301.3683229685*cos(theta)**3 - 121.350136399269*cos(theta))*sin(phi)

@torch.jit.script
def Yl28_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 60684770.0668697*cos(theta)**28 - 417069874.277759*cos(theta)**26 + 1278751973.02143*cos(theta)**24 - 2306768265.05827*cos(theta)**22 + 2718691169.53296*cos(theta)**20 - 2198090732.81388*cos(theta)**18 + 1245584748.59453*cos(theta)**16 - 496578637.313435*cos(theta)**14 + 137770292.669276*cos(theta)**12 - 25905525.1172998*cos(theta)**10 + 3150671.97372565*cos(theta)**8 - 229139.77990732*cos(theta)**6 + 8679.53711770151*cos(theta)**4 - 129.223877682901*cos(theta)**2 + 0.318285413012071

@torch.jit.script
def Yl28_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.105698659387677*(1.0 - cos(theta)**2)**0.5*(797820600.946591*cos(theta)**27 - 5091546016.95007*cos(theta)**25 + 14410035897.0285*cos(theta)**23 - 23828360012.7334*cos(theta)**21 + 25530385727.9286*cos(theta)**19 - 18577429614.7906*cos(theta)**17 + 9357520102.2649*cos(theta)**15 - 3264251198.4645*cos(theta)**13 + 776254858.171436*cos(theta)**11 - 121635376.63655*cos(theta)**9 + 11834793.4024751*cos(theta)**7 - 645534.185589552*cos(theta)**5 + 16301.3683229685*cos(theta)**3 - 121.350136399269*cos(theta))*cos(phi)

@torch.jit.script
def Yl28_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00371387232545999*(1.0 - cos(theta)**2)*(21541156225.558*cos(theta)**26 - 127288650423.752*cos(theta)**24 + 331430825631.655*cos(theta)**22 - 500395560267.401*cos(theta)**20 + 485077328830.644*cos(theta)**18 - 315816303451.44*cos(theta)**16 + 140362801533.974*cos(theta)**14 - 42435265580.0385*cos(theta)**12 + 8538803439.8858*cos(theta)**10 - 1094718389.72895*cos(theta)**8 + 82843553.8173258*cos(theta)**6 - 3227670.92794776*cos(theta)**4 + 48904.1049689054*cos(theta)**2 - 121.350136399269)*cos(2*phi)

@torch.jit.script
def Yl28_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000130815573253833*(1.0 - cos(theta)**2)**1.5*(560070061864.507*cos(theta)**25 - 3054927610170.04*cos(theta)**23 + 7291478163896.42*cos(theta)**21 - 10007911205348.0*cos(theta)**19 + 8731391918951.59*cos(theta)**17 - 5053060855223.05*cos(theta)**15 + 1965079221475.63*cos(theta)**13 - 509223186960.462*cos(theta)**11 + 85388034398.858*cos(theta)**9 - 8757747117.83159*cos(theta)**7 + 497061322.903955*cos(theta)**5 - 12910683.711791*cos(theta)**3 + 97808.2099378109*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl28_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.62502894662956e-6*(1.0 - cos(theta)**2)**2*(14001751546612.7*cos(theta)**24 - 70263335033910.9*cos(theta)**22 + 153121041441825.0*cos(theta)**20 - 190150312901612.0*cos(theta)**18 + 148433662622177.0*cos(theta)**16 - 75795912828345.7*cos(theta)**14 + 25546029879183.2*cos(theta)**12 - 5601455056565.08*cos(theta)**10 + 768492309589.722*cos(theta)**8 - 61304229824.8211*cos(theta)**6 + 2485306614.51977*cos(theta)**4 - 38732051.1353731*cos(theta)**2 + 97808.2099378109)*cos(4*phi)

@torch.jit.script
def Yl28_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.64343247431143e-7*(1.0 - cos(theta)**2)**2.5*(336042037118704.0*cos(theta)**23 - 1.54579337074604e+15*cos(theta)**21 + 3.06242082883649e+15*cos(theta)**19 - 3.42270563222902e+15*cos(theta)**17 + 2.37493860195483e+15*cos(theta)**15 - 1.06114277959684e+15*cos(theta)**13 + 306552358550198.0*cos(theta)**11 - 56014550565650.8*cos(theta)**9 + 6147938476717.77*cos(theta)**7 - 367825378948.927*cos(theta)**5 + 9941226458.0791*cos(theta)**3 - 77464102.2707462*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl28_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.8769025298657e-9*(1.0 - cos(theta)**2)**3*(7.7289668537302e+15*cos(theta)**22 - 3.24616607856668e+16*cos(theta)**20 + 5.81859957478934e+16*cos(theta)**18 - 5.81859957478934e+16*cos(theta)**16 + 3.56240790293225e+16*cos(theta)**14 - 1.37948561347589e+16*cos(theta)**12 + 3.37207594405218e+15*cos(theta)**10 - 504130955090858.0*cos(theta)**8 + 43035569337024.4*cos(theta)**6 - 1839126894744.63*cos(theta)**4 + 29823679374.2373*cos(theta)**2 - 77464102.2707462)*cos(6*phi)

@torch.jit.script
def Yl28_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.11788866150653e-10*(1.0 - cos(theta)**2)**3.5*(1.70037270782064e+17*cos(theta)**21 - 6.49233215713337e+17*cos(theta)**19 + 1.04734792346208e+18*cos(theta)**17 - 9.30975931966294e+17*cos(theta)**15 + 4.98737106410515e+17*cos(theta)**13 - 1.65538273617107e+17*cos(theta)**11 + 3.37207594405218e+16*cos(theta)**9 - 4.03304764072686e+15*cos(theta)**7 + 258213416022147.0*cos(theta)**5 - 7356507578978.53*cos(theta)**3 + 59647358748.4746*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl28_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.70268659114473e-12*(1.0 - cos(theta)**2)**4*(3.57078268642335e+18*cos(theta)**20 - 1.23354310985534e+19*cos(theta)**18 + 1.78049146988554e+19*cos(theta)**16 - 1.39646389794944e+19*cos(theta)**14 + 6.48358238333669e+18*cos(theta)**12 - 1.82092100978818e+18*cos(theta)**10 + 3.03486834964696e+17*cos(theta)**8 - 2.8231333485088e+16*cos(theta)**6 + 1.29106708011073e+15*cos(theta)**4 - 22069522736935.6*cos(theta)**2 + 59647358748.4746)*cos(8*phi)

@torch.jit.script
def Yl28_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.83156390560776e-13*(1.0 - cos(theta)**2)**4.5*(7.1415653728467e+19*cos(theta)**19 - 2.22037759773961e+20*cos(theta)**17 + 2.84878635181686e+20*cos(theta)**15 - 1.95504945712922e+20*cos(theta)**13 + 7.78029886000403e+19*cos(theta)**11 - 1.82092100978818e+19*cos(theta)**9 + 2.42789467971757e+18*cos(theta)**7 - 1.69388000910528e+17*cos(theta)**5 + 5.16426832044293e+15*cos(theta)**3 - 44139045473871.2*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl28_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.05379896790437e-14*(1.0 - cos(theta)**2)**5*(1.35689742084087e+21*cos(theta)**18 - 3.77464191615734e+21*cos(theta)**16 + 4.27317952772529e+21*cos(theta)**14 - 2.54156429426798e+21*cos(theta)**12 + 8.55832874600443e+20*cos(theta)**10 - 1.63882890880936e+20*cos(theta)**8 + 1.6995262758023e+19*cos(theta)**6 - 8.46940004552641e+17*cos(theta)**4 + 1.54928049613288e+16*cos(theta)**2 - 44139045473871.2)*cos(10*phi)

@torch.jit.script
def Yl28_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.977307899878e-16*(1.0 - cos(theta)**2)**5.5*(2.44241535751357e+22*cos(theta)**17 - 6.03942706585174e+22*cos(theta)**15 + 5.98245133881541e+22*cos(theta)**13 - 3.04987715312158e+22*cos(theta)**11 + 8.55832874600443e+21*cos(theta)**9 - 1.31106312704749e+21*cos(theta)**7 + 1.01971576548138e+20*cos(theta)**5 - 3.38776001821056e+18*cos(theta)**3 + 3.09856099226576e+16*cos(theta))*cos(11*phi)

@torch.jit.script
def Yl28_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.52522795453625e-17*(1.0 - cos(theta)**2)**6*(4.15210610777307e+23*cos(theta)**16 - 9.05914059877762e+23*cos(theta)**14 + 7.77718674046003e+23*cos(theta)**12 - 3.35486486843374e+23*cos(theta)**10 + 7.70249587140399e+22*cos(theta)**8 - 9.17744188933241e+21*cos(theta)**6 + 5.0985788274069e+20*cos(theta)**4 - 1.01632800546317e+19*cos(theta)**2 + 3.09856099226576e+16)*cos(12*phi)

@torch.jit.script
def Yl28_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.95501468493973e-19*(1.0 - cos(theta)**2)**6.5*(6.64336977243692e+24*cos(theta)**15 - 1.26827968382887e+25*cos(theta)**13 + 9.33262408855204e+24*cos(theta)**11 - 3.35486486843374e+24*cos(theta)**9 + 6.16199669712319e+23*cos(theta)**7 - 5.50646513359945e+22*cos(theta)**5 + 2.03943153096276e+21*cos(theta)**3 - 2.03265601092634e+19*cos(theta))*cos(13*phi)

@torch.jit.script
def Yl28_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.3725346401488e-20*(1.0 - cos(theta)**2)**7*(9.96505465865538e+25*cos(theta)**14 - 1.64876358897753e+26*cos(theta)**12 + 1.02658864974072e+26*cos(theta)**10 - 3.01937838159036e+25*cos(theta)**8 + 4.31339768798623e+24*cos(theta)**6 - 2.75323256679972e+23*cos(theta)**4 + 6.11829459288828e+21*cos(theta)**2 - 2.03265601092634e+19)*cos(14*phi)

@torch.jit.script
def Yl28_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.66972930141058e-22*(1.0 - cos(theta)**2)**7.5*(1.39510765221175e+27*cos(theta)**13 - 1.97851630677303e+27*cos(theta)**11 + 1.02658864974072e+27*cos(theta)**9 - 2.41550270527229e+26*cos(theta)**7 + 2.58803861279174e+25*cos(theta)**5 - 1.10129302671989e+24*cos(theta)**3 + 1.22365891857766e+22*cos(theta))*cos(15*phi)

@torch.jit.script
def Yl28_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.04311693361802e-23*(1.0 - cos(theta)**2)**8*(1.81363994787528e+28*cos(theta)**12 - 2.17636793745033e+28*cos(theta)**10 + 9.23929784766651e+27*cos(theta)**8 - 1.6908518936906e+27*cos(theta)**6 + 1.29401930639587e+26*cos(theta)**4 - 3.30387908015967e+24*cos(theta)**2 + 1.22365891857766e+22)*cos(16*phi)

@torch.jit.script
def Yl28_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.7398805056302e-24*(1.0 - cos(theta)**2)**8.5*(2.17636793745033e+29*cos(theta)**11 - 2.17636793745033e+29*cos(theta)**9 + 7.39143827813321e+28*cos(theta)**7 - 1.01451113621436e+28*cos(theta)**5 + 5.17607722558348e+26*cos(theta)**3 - 6.60775816031934e+24*cos(theta))*cos(17*phi)

@torch.jit.script
def Yl28_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.73471228858538e-26*(1.0 - cos(theta)**2)**9*(2.39400473119537e+30*cos(theta)**10 - 1.9587311437053e+30*cos(theta)**8 + 5.17400679469325e+29*cos(theta)**6 - 5.07255568107181e+28*cos(theta)**4 + 1.55282316767504e+27*cos(theta)**2 - 6.60775816031934e+24)*cos(18*phi)

@torch.jit.script
def Yl28_m19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.56775673567227e-27*(1.0 - cos(theta)**2)**9.5*(2.39400473119537e+31*cos(theta)**9 - 1.56698491496424e+31*cos(theta)**7 + 3.10440407681595e+30*cos(theta)**5 - 2.02902227242872e+29*cos(theta)**3 + 3.10564633535009e+27*cos(theta))*cos(19*phi)

@torch.jit.script
def Yl28_m20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.71653775978624e-28*(1.0 - cos(theta)**2)**10*(2.15460425807583e+32*cos(theta)**8 - 1.09688944047497e+32*cos(theta)**6 + 1.55220203840797e+31*cos(theta)**4 - 6.08706681728617e+29*cos(theta)**2 + 3.10564633535009e+27)*cos(20*phi)

@torch.jit.script
def Yl28_m21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.66982492934009e-30*(1.0 - cos(theta)**2)**10.5*(1.72368340646067e+33*cos(theta)**7 - 6.58133664284981e+32*cos(theta)**5 + 6.2088081536319e+31*cos(theta)**3 - 1.21741336345723e+30*cos(theta))*cos(21*phi)

@torch.jit.script
def Yl28_m22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.63421635555746e-31*(1.0 - cos(theta)**2)**11*(1.20657838452247e+34*cos(theta)**6 - 3.29066832142491e+33*cos(theta)**4 + 1.86264244608957e+32*cos(theta)**2 - 1.21741336345723e+30)*cos(22*phi)

@torch.jit.script
def Yl28_m23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.64920516074126e-32*(1.0 - cos(theta)**2)**11.5*(7.23947030713479e+34*cos(theta)**5 - 1.31626732856996e+34*cos(theta)**3 + 3.72528489217914e+32*cos(theta))*cos(23*phi)

@torch.jit.script
def Yl28_m24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.64296729492452e-33*(1.0 - cos(theta)**2)**12*(3.6197351535674e+35*cos(theta)**4 - 3.94880198570989e+34*cos(theta)**2 + 3.72528489217914e+32)*cos(24*phi)

@torch.jit.script
def Yl28_m25(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.12839457090042e-34*(1.0 - cos(theta)**2)**12.5*(1.44789406142696e+36*cos(theta)**3 - 7.89760397141977e+34*cos(theta))*cos(25*phi)

@torch.jit.script
def Yl28_m26(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.86550503264189e-36*(1.0 - cos(theta)**2)**13*(4.34368218428088e+36*cos(theta)**2 - 7.89760397141977e+34)*cos(26*phi)

@torch.jit.script
def Yl28_m27(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.34336601605245*(1.0 - cos(theta)**2)**13.5*cos(27*phi)*cos(theta)

@torch.jit.script
def Yl28_m28(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.981298560633835*(1.0 - cos(theta)**2)**14*cos(28*phi)

@torch.jit.script
def Yl29_m_minus_29(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.989721878741179*(1.0 - cos(theta)**2)**14.5*sin(29*phi)

@torch.jit.script
def Yl29_m_minus_28(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.53749726640217*(1.0 - cos(theta)**2)**14*sin(28*phi)*cos(theta)

@torch.jit.script
def Yl29_m_minus_27(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.62523699825355e-37*(1.0 - cos(theta)**2)**13.5*(2.4758988450401e+38*cos(theta)**2 - 4.34368218428088e+36)*sin(27*phi)

@torch.jit.script
def Yl29_m_minus_26(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.106547911828e-36*(1.0 - cos(theta)**2)**13*(8.25299615013366e+37*cos(theta)**3 - 4.34368218428088e+36*cos(theta))*sin(26*phi)

@torch.jit.script
def Yl29_m_minus_25(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.12451548733867e-35*(1.0 - cos(theta)**2)**12.5*(2.06324903753342e+37*cos(theta)**4 - 2.17184109214044e+36*cos(theta)**2 + 1.97440099285494e+34)*sin(25*phi)

@torch.jit.script
def Yl29_m_minus_24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.13410284106891e-34*(1.0 - cos(theta)**2)**12*(4.12649807506683e+36*cos(theta)**5 - 7.23947030713479e+35*cos(theta)**3 + 1.97440099285494e+34*cos(theta))*sin(24*phi)

@torch.jit.script
def Yl29_m_minus_23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.15541687226183e-33*(1.0 - cos(theta)**2)**11.5*(6.87749679177805e+35*cos(theta)**6 - 1.8098675767837e+35*cos(theta)**4 + 9.87200496427472e+33*cos(theta)**2 - 6.2088081536319e+31)*sin(23*phi)

@torch.jit.script
def Yl29_m_minus_22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.74674221195294e-31*(1.0 - cos(theta)**2)**11*(9.82499541682579e+34*cos(theta)**7 - 3.6197351535674e+34*cos(theta)**5 + 3.29066832142491e+33*cos(theta)**3 - 6.2088081536319e+31*cos(theta))*sin(22*phi)

@torch.jit.script
def Yl29_m_minus_21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.52824631913283e-30*(1.0 - cos(theta)**2)**10.5*(1.22812442710322e+34*cos(theta)**8 - 6.03289192261233e+33*cos(theta)**6 + 8.22667080356226e+32*cos(theta)**4 - 3.10440407681595e+31*cos(theta)**2 + 1.52176670432154e+29)*sin(21*phi)

@torch.jit.script
def Yl29_m_minus_20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.48454069386591e-29*(1.0 - cos(theta)**2)**10*(1.36458269678136e+33*cos(theta)**9 - 8.61841703230333e+32*cos(theta)**7 + 1.64533416071245e+32*cos(theta)**5 - 1.03480135893865e+31*cos(theta)**3 + 1.52176670432154e+29*cos(theta))*sin(20*phi)

@torch.jit.script
def Yl29_m_minus_19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.65677370829833e-27*(1.0 - cos(theta)**2)**9.5*(1.36458269678136e+32*cos(theta)**10 - 1.07730212903792e+32*cos(theta)**8 + 2.74222360118742e+31*cos(theta)**6 - 2.58700339734662e+30*cos(theta)**4 + 7.60883352160772e+28*cos(theta)**2 - 3.10564633535009e+26)*sin(19*phi)

@torch.jit.script
def Yl29_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.80697614338276e-26*(1.0 - cos(theta)**2)**9*(1.24052972434669e+31*cos(theta)**11 - 1.19700236559768e+31*cos(theta)**9 + 3.9174622874106e+30*cos(theta)**7 - 5.17400679469325e+29*cos(theta)**5 + 2.53627784053591e+28*cos(theta)**3 - 3.10564633535009e+26*cos(theta))*sin(18*phi)

@torch.jit.script
def Yl29_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.04106740874383e-25*(1.0 - cos(theta)**2)**8.5*(1.03377477028891e+30*cos(theta)**12 - 1.19700236559768e+30*cos(theta)**10 + 4.89682785926325e+29*cos(theta)**8 - 8.62334465782208e+28*cos(theta)**6 + 6.34069460133976e+27*cos(theta)**4 - 1.55282316767504e+26*cos(theta)**2 + 5.50646513359945e+23)*sin(17*phi)

@torch.jit.script
def Yl29_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.21090610686865e-23*(1.0 - cos(theta)**2)**8*(7.95211361760699e+28*cos(theta)**13 - 1.08818396872517e+29*cos(theta)**11 + 5.44091984362584e+28*cos(theta)**9 - 1.23190637968887e+28*cos(theta)**7 + 1.26813892026795e+27*cos(theta)**5 - 5.17607722558348e+25*cos(theta)**3 + 5.50646513359945e+23*cos(theta))*sin(16*phi)

@torch.jit.script
def Yl29_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.54933028611123e-22*(1.0 - cos(theta)**2)**7.5*(5.68008115543357e+27*cos(theta)**14 - 9.06819973937639e+27*cos(theta)**12 + 5.44091984362584e+27*cos(theta)**10 - 1.53988297461109e+27*cos(theta)**8 + 2.11356486711325e+26*cos(theta)**6 - 1.29401930639587e+25*cos(theta)**4 + 2.75323256679972e+23*cos(theta)**2 - 8.74042084698325e+20)*sin(15*phi)

@torch.jit.script
def Yl29_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.42564876361858e-20*(1.0 - cos(theta)**2)**7*(3.78672077028904e+26*cos(theta)**15 - 6.97553826105876e+26*cos(theta)**13 + 4.94629076693258e+26*cos(theta)**11 - 1.71098108290121e+26*cos(theta)**9 + 3.01937838159036e+25*cos(theta)**7 - 2.58803861279174e+24*cos(theta)**5 + 9.17744188933241e+22*cos(theta)**3 - 8.74042084698325e+20*cos(theta))*sin(14*phi)

@torch.jit.script
def Yl29_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.7394416498704e-19*(1.0 - cos(theta)**2)**6.5*(2.36670048143065e+25*cos(theta)**16 - 4.98252732932769e+25*cos(theta)**14 + 4.12190897244381e+25*cos(theta)**12 - 1.71098108290121e+25*cos(theta)**10 + 3.77422297698796e+24*cos(theta)**8 - 4.31339768798623e+23*cos(theta)**6 + 2.2943604723331e+22*cos(theta)**4 - 4.37021042349163e+20*cos(theta)**2 + 1.27041000682896e+18)*sin(13*phi)

@torch.jit.script
def Yl29_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.99207917847372e-18*(1.0 - cos(theta)**2)**6*(1.39217675378274e+24*cos(theta)**17 - 3.32168488621846e+24*cos(theta)**15 + 3.17069920957217e+24*cos(theta)**13 - 1.55543734809201e+24*cos(theta)**11 + 4.19358108554217e+23*cos(theta)**9 - 6.16199669712319e+22*cos(theta)**7 + 4.58872094466621e+21*cos(theta)**5 - 1.45673680783054e+20*cos(theta)**3 + 1.27041000682896e+18*cos(theta))*sin(12*phi)

@torch.jit.script
def Yl29_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.7144637587553e-16*(1.0 - cos(theta)**2)**5.5*(7.73431529879298e+22*cos(theta)**18 - 2.07605305388654e+23*cos(theta)**16 + 2.2647851496944e+23*cos(theta)**14 - 1.29619779007667e+23*cos(theta)**12 + 4.19358108554217e+22*cos(theta)**10 - 7.70249587140399e+21*cos(theta)**8 + 7.64786824111034e+20*cos(theta)**6 - 3.64184201957635e+19*cos(theta)**4 + 6.35205003414481e+17*cos(theta)**2 - 1.72142277348098e+15)*sin(11*phi)

@torch.jit.script
def Yl29_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.48326015729302e-15*(1.0 - cos(theta)**2)**5*(4.07069226252262e+21*cos(theta)**19 - 1.22120767875679e+22*cos(theta)**17 + 1.50985676646294e+22*cos(theta)**15 - 9.97075223135901e+21*cos(theta)**13 + 3.81234644140198e+21*cos(theta)**11 - 8.55832874600443e+20*cos(theta)**9 + 1.09255260587291e+20*cos(theta)**7 - 7.28368403915271e+18*cos(theta)**5 + 2.1173500113816e+17*cos(theta)**3 - 1.72142277348098e+15*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl29_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.08996082292824e-13*(1.0 - cos(theta)**2)**4.5*(2.03534613126131e+20*cos(theta)**20 - 6.78448710420437e+20*cos(theta)**18 + 9.43660479039335e+20*cos(theta)**16 - 7.12196587954215e+20*cos(theta)**14 + 3.17695536783498e+20*cos(theta)**12 - 8.55832874600443e+19*cos(theta)**10 + 1.36569075734113e+19*cos(theta)**8 - 1.21394733985879e+18*cos(theta)**6 + 5.293375028454e+16*cos(theta)**4 - 860711386740489.0*cos(theta)**2 + 2206952273693.56)*sin(9*phi)

@torch.jit.script
def Yl29_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.90390812988918e-12*(1.0 - cos(theta)**2)**4*(9.69212443457767e+18*cos(theta)**21 - 3.57078268642335e+19*cos(theta)**19 + 5.55094399434903e+19*cos(theta)**17 - 4.7479772530281e+19*cos(theta)**15 + 2.44381182141152e+19*cos(theta)**13 - 7.78029886000403e+18*cos(theta)**11 + 1.51743417482348e+18*cos(theta)**9 - 1.73421048551255e+17*cos(theta)**7 + 1.0586750056908e+16*cos(theta)**5 - 286903795580163.0*cos(theta)**3 + 2206952273693.56*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl29_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.68442544512435e-10*(1.0 - cos(theta)**2)**3.5*(4.40551110662621e+17*cos(theta)**22 - 1.78539134321168e+18*cos(theta)**20 + 3.08385777463835e+18*cos(theta)**18 - 2.96748578314256e+18*cos(theta)**16 + 1.7455798724368e+18*cos(theta)**14 - 6.48358238333669e+17*cos(theta)**12 + 1.51743417482348e+17*cos(theta)**10 - 2.16776310689069e+16*cos(theta)**8 + 1.764458342818e+15*cos(theta)**6 - 71725948895040.7*cos(theta)**4 + 1103476136846.78*cos(theta)**2 - 2711243579.47612)*sin(7*phi)

@torch.jit.script
def Yl29_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.84693238903845e-9*(1.0 - cos(theta)**2)**3*(1.91543961157661e+16*cos(theta)**23 - 8.50186353910322e+16*cos(theta)**21 + 1.62308303928334e+17*cos(theta)**19 - 1.7455798724368e+17*cos(theta)**17 + 1.16371991495787e+17*cos(theta)**15 - 4.98737106410515e+16*cos(theta)**13 + 1.37948561347589e+16*cos(theta)**11 - 2.40862567432299e+15*cos(theta)**9 + 252065477545429.0*cos(theta)**7 - 14345189779008.1*cos(theta)**5 + 367825378948.927*cos(theta)**3 - 2711243579.47612*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl29_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.40477446625728e-7*(1.0 - cos(theta)**2)**2.5*(798099838156923.0*cos(theta)**24 - 3.8644834268651e+15*cos(theta)**22 + 8.11541519641671e+15*cos(theta)**20 - 9.69766595798223e+15*cos(theta)**18 + 7.27324946848667e+15*cos(theta)**16 - 3.56240790293225e+15*cos(theta)**14 + 1.14957134456324e+15*cos(theta)**12 - 240862567432299.0*cos(theta)**10 + 31508184693178.6*cos(theta)**8 - 2390864963168.02*cos(theta)**6 + 91956344737.2317*cos(theta)**4 - 1355621789.73806*cos(theta)**2 + 3227670.92794776)*sin(5*phi)

@torch.jit.script
def Yl29_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.0955861679266e-6*(1.0 - cos(theta)**2)**2*(31923993526276.9*cos(theta)**25 - 168021018559352.0*cos(theta)**23 + 386448342686510.0*cos(theta)**21 - 510403471472749.0*cos(theta)**19 + 427838204028628.0*cos(theta)**17 - 237493860195483.0*cos(theta)**15 + 88428564966403.3*cos(theta)**13 - 21896597039299.9*cos(theta)**11 + 3500909410353.18*cos(theta)**9 - 341552137595.432*cos(theta)**7 + 18391268947.4463*cos(theta)**5 - 451873929.912686*cos(theta)**3 + 3227670.92794776*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl29_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000119966423463177*(1.0 - cos(theta)**2)**1.5*(1227845904856.8*cos(theta)**26 - 7000875773306.34*cos(theta)**24 + 17565833758477.7*cos(theta)**22 - 25520173573637.5*cos(theta)**20 + 23768789112701.5*cos(theta)**18 - 14843366262217.7*cos(theta)**16 + 6316326069028.81*cos(theta)**14 - 1824716419941.66*cos(theta)**12 + 350090941035.318*cos(theta)**10 - 42694017199.429*cos(theta)**8 + 3065211491.24106*cos(theta)**6 - 112968482.478172*cos(theta)**4 + 1613835.46397388*cos(theta)**2 - 3761.85422837734)*sin(3*phi)

@torch.jit.script
def Yl29_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00352627828501722*(1.0 - cos(theta)**2)*(45475774253.9557*cos(theta)**27 - 280035030932.254*cos(theta)**25 + 763731902542.51*cos(theta)**23 - 1215246360649.4*cos(theta)**21 + 1250988900668.5*cos(theta)**19 - 873139191895.159*cos(theta)**17 + 421088404601.921*cos(theta)**15 - 140362801533.974*cos(theta)**13 + 31826449185.0289*cos(theta)**11 - 4743779688.82544*cos(theta)**9 + 437887355.891579*cos(theta)**7 - 22593696.4956343*cos(theta)**5 + 537945.15465796*cos(theta)**3 - 3761.85422837734*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl29_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.103890645660027*(1.0 - cos(theta)**2)**0.5*(1624134794.78413*cos(theta)**28 - 10770578112.779*cos(theta)**26 + 31822162605.9379*cos(theta)**24 - 55238470938.6092*cos(theta)**22 + 62549445033.4251*cos(theta)**20 - 48507732883.0644*cos(theta)**18 + 26318025287.62*cos(theta)**16 - 10025914395.2838*cos(theta)**14 + 2652204098.75241*cos(theta)**12 - 474377968.882544*cos(theta)**10 + 54735919.4864474*cos(theta)**8 - 3765616.08260572*cos(theta)**6 + 134486.28866449*cos(theta)**4 - 1880.92711418867*cos(theta)**2 + 4.33393344283104)*sin(phi)

@torch.jit.script
def Yl29_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 121351499.324998*cos(theta)**29 - 864363310.981568*cos(theta)**27 + 2758104746.85937*cos(theta)**25 - 5203971220.48937*cos(theta)**23 + 6453944699.92064*cos(theta)**21 - 5531952599.93198*cos(theta)**19 + 3354481895.70343*cos(theta)**17 - 1448284247.03386*cos(theta)**15 + 442063505.635336*cos(theta)**13 - 93444318.26438*cos(theta)**11 + 13178044.8834382*cos(theta)**9 - 1165625.59165547*cos(theta)**7 + 58281.2795827734*cos(theta)**5 - 1358.53798561243*cos(theta)**3 + 9.39081556874954*cos(theta)

@torch.jit.script
def Yl29_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.103890645660027*(1.0 - cos(theta)**2)**0.5*(1624134794.78413*cos(theta)**28 - 10770578112.779*cos(theta)**26 + 31822162605.9379*cos(theta)**24 - 55238470938.6092*cos(theta)**22 + 62549445033.4251*cos(theta)**20 - 48507732883.0644*cos(theta)**18 + 26318025287.62*cos(theta)**16 - 10025914395.2838*cos(theta)**14 + 2652204098.75241*cos(theta)**12 - 474377968.882544*cos(theta)**10 + 54735919.4864474*cos(theta)**8 - 3765616.08260572*cos(theta)**6 + 134486.28866449*cos(theta)**4 - 1880.92711418867*cos(theta)**2 + 4.33393344283104)*cos(phi)

@torch.jit.script
def Yl29_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00352627828501722*(1.0 - cos(theta)**2)*(45475774253.9557*cos(theta)**27 - 280035030932.254*cos(theta)**25 + 763731902542.51*cos(theta)**23 - 1215246360649.4*cos(theta)**21 + 1250988900668.5*cos(theta)**19 - 873139191895.159*cos(theta)**17 + 421088404601.921*cos(theta)**15 - 140362801533.974*cos(theta)**13 + 31826449185.0289*cos(theta)**11 - 4743779688.82544*cos(theta)**9 + 437887355.891579*cos(theta)**7 - 22593696.4956343*cos(theta)**5 + 537945.15465796*cos(theta)**3 - 3761.85422837734*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl29_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000119966423463177*(1.0 - cos(theta)**2)**1.5*(1227845904856.8*cos(theta)**26 - 7000875773306.34*cos(theta)**24 + 17565833758477.7*cos(theta)**22 - 25520173573637.5*cos(theta)**20 + 23768789112701.5*cos(theta)**18 - 14843366262217.7*cos(theta)**16 + 6316326069028.81*cos(theta)**14 - 1824716419941.66*cos(theta)**12 + 350090941035.318*cos(theta)**10 - 42694017199.429*cos(theta)**8 + 3065211491.24106*cos(theta)**6 - 112968482.478172*cos(theta)**4 + 1613835.46397388*cos(theta)**2 - 3761.85422837734)*cos(3*phi)

@torch.jit.script
def Yl29_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.0955861679266e-6*(1.0 - cos(theta)**2)**2*(31923993526276.9*cos(theta)**25 - 168021018559352.0*cos(theta)**23 + 386448342686510.0*cos(theta)**21 - 510403471472749.0*cos(theta)**19 + 427838204028628.0*cos(theta)**17 - 237493860195483.0*cos(theta)**15 + 88428564966403.3*cos(theta)**13 - 21896597039299.9*cos(theta)**11 + 3500909410353.18*cos(theta)**9 - 341552137595.432*cos(theta)**7 + 18391268947.4463*cos(theta)**5 - 451873929.912686*cos(theta)**3 + 3227670.92794776*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl29_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.40477446625728e-7*(1.0 - cos(theta)**2)**2.5*(798099838156923.0*cos(theta)**24 - 3.8644834268651e+15*cos(theta)**22 + 8.11541519641671e+15*cos(theta)**20 - 9.69766595798223e+15*cos(theta)**18 + 7.27324946848667e+15*cos(theta)**16 - 3.56240790293225e+15*cos(theta)**14 + 1.14957134456324e+15*cos(theta)**12 - 240862567432299.0*cos(theta)**10 + 31508184693178.6*cos(theta)**8 - 2390864963168.02*cos(theta)**6 + 91956344737.2317*cos(theta)**4 - 1355621789.73806*cos(theta)**2 + 3227670.92794776)*cos(5*phi)

@torch.jit.script
def Yl29_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.84693238903845e-9*(1.0 - cos(theta)**2)**3*(1.91543961157661e+16*cos(theta)**23 - 8.50186353910322e+16*cos(theta)**21 + 1.62308303928334e+17*cos(theta)**19 - 1.7455798724368e+17*cos(theta)**17 + 1.16371991495787e+17*cos(theta)**15 - 4.98737106410515e+16*cos(theta)**13 + 1.37948561347589e+16*cos(theta)**11 - 2.40862567432299e+15*cos(theta)**9 + 252065477545429.0*cos(theta)**7 - 14345189779008.1*cos(theta)**5 + 367825378948.927*cos(theta)**3 - 2711243579.47612*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl29_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.68442544512435e-10*(1.0 - cos(theta)**2)**3.5*(4.40551110662621e+17*cos(theta)**22 - 1.78539134321168e+18*cos(theta)**20 + 3.08385777463835e+18*cos(theta)**18 - 2.96748578314256e+18*cos(theta)**16 + 1.7455798724368e+18*cos(theta)**14 - 6.48358238333669e+17*cos(theta)**12 + 1.51743417482348e+17*cos(theta)**10 - 2.16776310689069e+16*cos(theta)**8 + 1.764458342818e+15*cos(theta)**6 - 71725948895040.7*cos(theta)**4 + 1103476136846.78*cos(theta)**2 - 2711243579.47612)*cos(7*phi)

@torch.jit.script
def Yl29_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.90390812988918e-12*(1.0 - cos(theta)**2)**4*(9.69212443457767e+18*cos(theta)**21 - 3.57078268642335e+19*cos(theta)**19 + 5.55094399434903e+19*cos(theta)**17 - 4.7479772530281e+19*cos(theta)**15 + 2.44381182141152e+19*cos(theta)**13 - 7.78029886000403e+18*cos(theta)**11 + 1.51743417482348e+18*cos(theta)**9 - 1.73421048551255e+17*cos(theta)**7 + 1.0586750056908e+16*cos(theta)**5 - 286903795580163.0*cos(theta)**3 + 2206952273693.56*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl29_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.08996082292824e-13*(1.0 - cos(theta)**2)**4.5*(2.03534613126131e+20*cos(theta)**20 - 6.78448710420437e+20*cos(theta)**18 + 9.43660479039335e+20*cos(theta)**16 - 7.12196587954215e+20*cos(theta)**14 + 3.17695536783498e+20*cos(theta)**12 - 8.55832874600443e+19*cos(theta)**10 + 1.36569075734113e+19*cos(theta)**8 - 1.21394733985879e+18*cos(theta)**6 + 5.293375028454e+16*cos(theta)**4 - 860711386740489.0*cos(theta)**2 + 2206952273693.56)*cos(9*phi)

@torch.jit.script
def Yl29_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.48326015729302e-15*(1.0 - cos(theta)**2)**5*(4.07069226252262e+21*cos(theta)**19 - 1.22120767875679e+22*cos(theta)**17 + 1.50985676646294e+22*cos(theta)**15 - 9.97075223135901e+21*cos(theta)**13 + 3.81234644140198e+21*cos(theta)**11 - 8.55832874600443e+20*cos(theta)**9 + 1.09255260587291e+20*cos(theta)**7 - 7.28368403915271e+18*cos(theta)**5 + 2.1173500113816e+17*cos(theta)**3 - 1.72142277348098e+15*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl29_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.7144637587553e-16*(1.0 - cos(theta)**2)**5.5*(7.73431529879298e+22*cos(theta)**18 - 2.07605305388654e+23*cos(theta)**16 + 2.2647851496944e+23*cos(theta)**14 - 1.29619779007667e+23*cos(theta)**12 + 4.19358108554217e+22*cos(theta)**10 - 7.70249587140399e+21*cos(theta)**8 + 7.64786824111034e+20*cos(theta)**6 - 3.64184201957635e+19*cos(theta)**4 + 6.35205003414481e+17*cos(theta)**2 - 1.72142277348098e+15)*cos(11*phi)

@torch.jit.script
def Yl29_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.99207917847372e-18*(1.0 - cos(theta)**2)**6*(1.39217675378274e+24*cos(theta)**17 - 3.32168488621846e+24*cos(theta)**15 + 3.17069920957217e+24*cos(theta)**13 - 1.55543734809201e+24*cos(theta)**11 + 4.19358108554217e+23*cos(theta)**9 - 6.16199669712319e+22*cos(theta)**7 + 4.58872094466621e+21*cos(theta)**5 - 1.45673680783054e+20*cos(theta)**3 + 1.27041000682896e+18*cos(theta))*cos(12*phi)

@torch.jit.script
def Yl29_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.7394416498704e-19*(1.0 - cos(theta)**2)**6.5*(2.36670048143065e+25*cos(theta)**16 - 4.98252732932769e+25*cos(theta)**14 + 4.12190897244381e+25*cos(theta)**12 - 1.71098108290121e+25*cos(theta)**10 + 3.77422297698796e+24*cos(theta)**8 - 4.31339768798623e+23*cos(theta)**6 + 2.2943604723331e+22*cos(theta)**4 - 4.37021042349163e+20*cos(theta)**2 + 1.27041000682896e+18)*cos(13*phi)

@torch.jit.script
def Yl29_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.42564876361858e-20*(1.0 - cos(theta)**2)**7*(3.78672077028904e+26*cos(theta)**15 - 6.97553826105876e+26*cos(theta)**13 + 4.94629076693258e+26*cos(theta)**11 - 1.71098108290121e+26*cos(theta)**9 + 3.01937838159036e+25*cos(theta)**7 - 2.58803861279174e+24*cos(theta)**5 + 9.17744188933241e+22*cos(theta)**3 - 8.74042084698325e+20*cos(theta))*cos(14*phi)

@torch.jit.script
def Yl29_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.54933028611123e-22*(1.0 - cos(theta)**2)**7.5*(5.68008115543357e+27*cos(theta)**14 - 9.06819973937639e+27*cos(theta)**12 + 5.44091984362584e+27*cos(theta)**10 - 1.53988297461109e+27*cos(theta)**8 + 2.11356486711325e+26*cos(theta)**6 - 1.29401930639587e+25*cos(theta)**4 + 2.75323256679972e+23*cos(theta)**2 - 8.74042084698325e+20)*cos(15*phi)

@torch.jit.script
def Yl29_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.21090610686865e-23*(1.0 - cos(theta)**2)**8*(7.95211361760699e+28*cos(theta)**13 - 1.08818396872517e+29*cos(theta)**11 + 5.44091984362584e+28*cos(theta)**9 - 1.23190637968887e+28*cos(theta)**7 + 1.26813892026795e+27*cos(theta)**5 - 5.17607722558348e+25*cos(theta)**3 + 5.50646513359945e+23*cos(theta))*cos(16*phi)

@torch.jit.script
def Yl29_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.04106740874383e-25*(1.0 - cos(theta)**2)**8.5*(1.03377477028891e+30*cos(theta)**12 - 1.19700236559768e+30*cos(theta)**10 + 4.89682785926325e+29*cos(theta)**8 - 8.62334465782208e+28*cos(theta)**6 + 6.34069460133976e+27*cos(theta)**4 - 1.55282316767504e+26*cos(theta)**2 + 5.50646513359945e+23)*cos(17*phi)

@torch.jit.script
def Yl29_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.80697614338276e-26*(1.0 - cos(theta)**2)**9*(1.24052972434669e+31*cos(theta)**11 - 1.19700236559768e+31*cos(theta)**9 + 3.9174622874106e+30*cos(theta)**7 - 5.17400679469325e+29*cos(theta)**5 + 2.53627784053591e+28*cos(theta)**3 - 3.10564633535009e+26*cos(theta))*cos(18*phi)

@torch.jit.script
def Yl29_m19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.65677370829833e-27*(1.0 - cos(theta)**2)**9.5*(1.36458269678136e+32*cos(theta)**10 - 1.07730212903792e+32*cos(theta)**8 + 2.74222360118742e+31*cos(theta)**6 - 2.58700339734662e+30*cos(theta)**4 + 7.60883352160772e+28*cos(theta)**2 - 3.10564633535009e+26)*cos(19*phi)

@torch.jit.script
def Yl29_m20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.48454069386591e-29*(1.0 - cos(theta)**2)**10*(1.36458269678136e+33*cos(theta)**9 - 8.61841703230333e+32*cos(theta)**7 + 1.64533416071245e+32*cos(theta)**5 - 1.03480135893865e+31*cos(theta)**3 + 1.52176670432154e+29*cos(theta))*cos(20*phi)

@torch.jit.script
def Yl29_m21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.52824631913283e-30*(1.0 - cos(theta)**2)**10.5*(1.22812442710322e+34*cos(theta)**8 - 6.03289192261233e+33*cos(theta)**6 + 8.22667080356226e+32*cos(theta)**4 - 3.10440407681595e+31*cos(theta)**2 + 1.52176670432154e+29)*cos(21*phi)

@torch.jit.script
def Yl29_m22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.74674221195294e-31*(1.0 - cos(theta)**2)**11*(9.82499541682579e+34*cos(theta)**7 - 3.6197351535674e+34*cos(theta)**5 + 3.29066832142491e+33*cos(theta)**3 - 6.2088081536319e+31*cos(theta))*cos(22*phi)

@torch.jit.script
def Yl29_m23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.15541687226183e-33*(1.0 - cos(theta)**2)**11.5*(6.87749679177805e+35*cos(theta)**6 - 1.8098675767837e+35*cos(theta)**4 + 9.87200496427472e+33*cos(theta)**2 - 6.2088081536319e+31)*cos(23*phi)

@torch.jit.script
def Yl29_m24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.13410284106891e-34*(1.0 - cos(theta)**2)**12*(4.12649807506683e+36*cos(theta)**5 - 7.23947030713479e+35*cos(theta)**3 + 1.97440099285494e+34*cos(theta))*cos(24*phi)

@torch.jit.script
def Yl29_m25(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.12451548733867e-35*(1.0 - cos(theta)**2)**12.5*(2.06324903753342e+37*cos(theta)**4 - 2.17184109214044e+36*cos(theta)**2 + 1.97440099285494e+34)*cos(25*phi)

@torch.jit.script
def Yl29_m26(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.106547911828e-36*(1.0 - cos(theta)**2)**13*(8.25299615013366e+37*cos(theta)**3 - 4.34368218428088e+36*cos(theta))*cos(26*phi)

@torch.jit.script
def Yl29_m27(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.62523699825355e-37*(1.0 - cos(theta)**2)**13.5*(2.4758988450401e+38*cos(theta)**2 - 4.34368218428088e+36)*cos(27*phi)

@torch.jit.script
def Yl29_m28(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.53749726640217*(1.0 - cos(theta)**2)**14*cos(28*phi)*cos(theta)

@torch.jit.script
def Yl29_m29(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.989721878741179*(1.0 - cos(theta)**2)**14.5*cos(29*phi)

@torch.jit.script
def Yl30_m_minus_30(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.997935479150139*(1.0 - cos(theta)**2)**15*sin(30*phi)

@torch.jit.script
def Yl30_m_minus_29(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.72997498267602*(1.0 - cos(theta)**2)**14.5*sin(29*phi)*cos(theta)

@torch.jit.script
def Yl30_m_minus_28(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.87411530575892e-39*(1.0 - cos(theta)**2)**14*(1.46078031857366e+40*cos(theta)**2 - 2.4758988450401e+38)*sin(28*phi)

@torch.jit.script
def Yl30_m_minus_27(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.79121847114987e-38*(1.0 - cos(theta)**2)**13.5*(4.86926772857886e+39*cos(theta)**3 - 2.4758988450401e+38*cos(theta))*sin(27*phi)

@torch.jit.script
def Yl30_m_minus_26(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.72461435302436e-37*(1.0 - cos(theta)**2)**13*(1.21731693214472e+39*cos(theta)**4 - 1.23794942252005e+38*cos(theta)**2 + 1.08592054607022e+36)*sin(26*phi)

@torch.jit.script
def Yl30_m_minus_25(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.57911199299742e-36*(1.0 - cos(theta)**2)**12.5*(2.43463386428943e+38*cos(theta)**5 - 4.12649807506683e+37*cos(theta)**3 + 1.08592054607022e+36*cos(theta))*sin(25*phi)

@torch.jit.script
def Yl30_m_minus_24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.74013210905229e-34*(1.0 - cos(theta)**2)**12*(4.05772310714905e+37*cos(theta)**6 - 1.03162451876671e+37*cos(theta)**4 + 5.42960273035109e+35*cos(theta)**2 - 3.29066832142491e+33)*sin(24*phi)

@torch.jit.script
def Yl30_m_minus_23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.38320349392245e-33*(1.0 - cos(theta)**2)**11.5*(5.79674729592722e+36*cos(theta)**7 - 2.06324903753342e+36*cos(theta)**5 + 1.8098675767837e+35*cos(theta)**3 - 3.29066832142491e+33*cos(theta))*sin(23*phi)

@torch.jit.script
def Yl30_m_minus_22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.96644237302409e-32*(1.0 - cos(theta)**2)**11*(7.24593411990902e+35*cos(theta)**8 - 3.43874839588903e+35*cos(theta)**6 + 4.52466894195925e+34*cos(theta)**4 - 1.64533416071245e+33*cos(theta)**2 + 7.76101019203987e+30)*sin(22*phi)

@torch.jit.script
def Yl30_m_minus_21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.5070719110102e-30*(1.0 - cos(theta)**2)**10.5*(8.05103791101002e+34*cos(theta)**9 - 4.9124977084129e+34*cos(theta)**7 + 9.04933788391849e+33*cos(theta)**5 - 5.48444720237484e+32*cos(theta)**3 + 7.76101019203987e+30*cos(theta))*sin(21*phi)

@torch.jit.script
def Yl30_m_minus_20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.40344756082348e-29*(1.0 - cos(theta)**2)**10*(8.05103791101002e+33*cos(theta)**10 - 6.14062213551612e+33*cos(theta)**8 + 1.50822298065308e+33*cos(theta)**6 - 1.37111180059371e+32*cos(theta)**4 + 3.88050509601994e+30*cos(theta)**2 - 1.52176670432154e+28)*sin(20*phi)

@torch.jit.script
def Yl30_m_minus_19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.98179203850954e-28*(1.0 - cos(theta)**2)**9.5*(7.31912537364547e+32*cos(theta)**11 - 6.8229134839068e+32*cos(theta)**9 + 2.15460425807583e+32*cos(theta)**7 - 2.74222360118742e+31*cos(theta)**5 + 1.29350169867331e+30*cos(theta)**3 - 1.52176670432154e+28*cos(theta))*sin(19*phi)

@torch.jit.script
def Yl30_m_minus_18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.93548170846062e-26*(1.0 - cos(theta)**2)**9*(6.09927114470456e+31*cos(theta)**12 - 6.8229134839068e+31*cos(theta)**10 + 2.69325532259479e+31*cos(theta)**8 - 4.5703726686457e+30*cos(theta)**6 + 3.23375424668328e+29*cos(theta)**4 - 7.60883352160772e+27*cos(theta)**2 + 2.58803861279174e+25)*sin(18*phi)

@torch.jit.script
def Yl30_m_minus_17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.83483175810931e-25*(1.0 - cos(theta)**2)**8.5*(4.69174703438813e+30*cos(theta)**13 - 6.20264862173345e+30*cos(theta)**11 + 2.99250591399421e+30*cos(theta)**9 - 6.529103812351e+29*cos(theta)**7 + 6.46750849336656e+28*cos(theta)**5 - 2.53627784053591e+27*cos(theta)**3 + 2.58803861279174e+25*cos(theta))*sin(17*phi)

@torch.jit.script
def Yl30_m_minus_16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.24020738463486e-23*(1.0 - cos(theta)**2)**8*(3.3512478817058e+29*cos(theta)**14 - 5.16887385144454e+29*cos(theta)**12 + 2.99250591399421e+29*cos(theta)**10 - 8.16137976543875e+28*cos(theta)**8 + 1.07791808222776e+28*cos(theta)**6 - 6.34069460133976e+26*cos(theta)**4 + 1.29401930639587e+25*cos(theta)**2 - 3.93318938114246e+22)*sin(16*phi)

@torch.jit.script
def Yl30_m_minus_15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.25775828793813e-22*(1.0 - cos(theta)**2)**7.5*(2.23416525447054e+28*cos(theta)**15 - 3.9760568088035e+28*cos(theta)**13 + 2.72045992181292e+28*cos(theta)**11 - 9.06819973937639e+27*cos(theta)**9 + 1.53988297461109e+27*cos(theta)**7 - 1.26813892026795e+26*cos(theta)**5 + 4.31339768798623e+24*cos(theta)**3 - 3.93318938114246e+22*cos(theta))*sin(15*phi)

@torch.jit.script
def Yl30_m_minus_14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.74148278331158e-21*(1.0 - cos(theta)**2)**7*(1.39635328404408e+27*cos(theta)**16 - 2.84004057771678e+27*cos(theta)**14 + 2.2670499348441e+27*cos(theta)**12 - 9.06819973937639e+26*cos(theta)**10 + 1.92485371826386e+26*cos(theta)**8 - 2.11356486711326e+25*cos(theta)**6 + 1.07834942199656e+24*cos(theta)**4 - 1.96659469057123e+22*cos(theta)**2 + 5.46276302936453e+19)*sin(14*phi)

@torch.jit.script
def Yl30_m_minus_13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.39075958422627e-19*(1.0 - cos(theta)**2)**6.5*(8.21384284731815e+25*cos(theta)**17 - 1.89336038514452e+26*cos(theta)**15 + 1.74388456526469e+26*cos(theta)**13 - 8.24381794488763e+25*cos(theta)**11 + 2.13872635362651e+25*cos(theta)**9 - 3.01937838159036e+24*cos(theta)**7 + 2.15669884399312e+23*cos(theta)**5 - 6.55531563523744e+21*cos(theta)**3 + 5.46276302936453e+19*cos(theta))*sin(13*phi)

@torch.jit.script
def Yl30_m_minus_12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.65129768956931e-18*(1.0 - cos(theta)**2)**6*(4.56324602628786e+24*cos(theta)**18 - 1.18335024071533e+25*cos(theta)**16 + 1.24563183233192e+25*cos(theta)**14 - 6.86984828740636e+24*cos(theta)**12 + 2.13872635362651e+24*cos(theta)**10 - 3.77422297698795e+23*cos(theta)**8 + 3.59449807332186e+22*cos(theta)**6 - 1.63882890880936e+21*cos(theta)**4 + 2.73138151468227e+19*cos(theta)**2 - 7.05783337127201e+16)*sin(12*phi)

@torch.jit.script
def Yl30_m_minus_11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.87891801956087e-16*(1.0 - cos(theta)**2)**5.5*(2.40170843488835e+23*cos(theta)**19 - 6.96088376891368e+23*cos(theta)**17 + 8.30421221554615e+23*cos(theta)**15 - 5.28449868262028e+23*cos(theta)**13 + 1.94429668511501e+23*cos(theta)**11 - 4.19358108554217e+22*cos(theta)**9 + 5.13499724760266e+21*cos(theta)**7 - 3.27765781761872e+20*cos(theta)**5 + 9.10460504894089e+18*cos(theta)**3 - 7.05783337127201e+16*cos(theta))*sin(11*phi)

@torch.jit.script
def Yl30_m_minus_10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.38040239932763e-15*(1.0 - cos(theta)**2)**5*(1.20085421744417e+22*cos(theta)**20 - 3.86715764939649e+22*cos(theta)**18 + 5.19013263471634e+22*cos(theta)**16 - 3.77464191615734e+22*cos(theta)**14 + 1.62024723759584e+22*cos(theta)**12 - 4.19358108554217e+21*cos(theta)**10 + 6.41874655950332e+20*cos(theta)**8 - 5.46276302936453e+19*cos(theta)**6 + 2.27615126223522e+18*cos(theta)**4 - 3.528916685636e+16*cos(theta)**2 + 86071138674048.8)*sin(10*phi)

@torch.jit.script
def Yl30_m_minus_9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.55938876429517e-13*(1.0 - cos(theta)**2)**4.5*(5.71835341640083e+20*cos(theta)**21 - 2.03534613126131e+21*cos(theta)**19 + 3.05301919689197e+21*cos(theta)**17 - 2.51642794410489e+21*cos(theta)**15 + 1.24634402891988e+21*cos(theta)**13 - 3.81234644140197e+20*cos(theta)**11 + 7.13194062167036e+19*cos(theta)**9 - 7.80394718480647e+18*cos(theta)**7 + 4.55230252447044e+17*cos(theta)**5 - 1.17630556187867e+16*cos(theta)**3 + 86071138674048.8*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl30_m_minus_8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.56770496751288e-12*(1.0 - cos(theta)**2)**4*(2.59925155290947e+19*cos(theta)**22 - 1.01767306563066e+20*cos(theta)**20 + 1.69612177605109e+20*cos(theta)**18 - 1.57276746506556e+20*cos(theta)**16 + 8.90245734942769e+19*cos(theta)**14 - 3.17695536783498e+19*cos(theta)**12 + 7.13194062167036e+18*cos(theta)**10 - 9.75493398100809e+17*cos(theta)**8 + 7.58717087411741e+16*cos(theta)**6 - 2.94076390469667e+15*cos(theta)**4 + 43035569337024.4*cos(theta)**2 - 100316012440.616)*sin(8*phi)

@torch.jit.script
def Yl30_m_minus_7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.3503730468945e-10*(1.0 - cos(theta)**2)**3.5*(1.1301093708302e+18*cos(theta)**23 - 4.84606221728884e+18*cos(theta)**21 + 8.92695671605838e+18*cos(theta)**19 - 9.25157332391505e+18*cos(theta)**17 + 5.93497156628513e+18*cos(theta)**15 - 2.44381182141152e+18*cos(theta)**13 + 6.48358238333669e+17*cos(theta)**11 - 1.08388155344534e+17*cos(theta)**9 + 1.08388155344534e+16*cos(theta)**7 - 588152780939334.0*cos(theta)**5 + 14345189779008.1*cos(theta)**3 - 100316012440.616*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl30_m_minus_6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.02402104966148e-9*(1.0 - cos(theta)**2)**3*(4.70878904512584e+16*cos(theta)**24 - 2.20275555331311e+17*cos(theta)**22 + 4.46347835802919e+17*cos(theta)**20 - 5.13976295773058e+17*cos(theta)**18 + 3.7093572289282e+17*cos(theta)**16 - 1.7455798724368e+17*cos(theta)**14 + 5.40298531944724e+16*cos(theta)**12 - 1.08388155344534e+16*cos(theta)**10 + 1.35485194180668e+15*cos(theta)**8 - 98025463489889.0*cos(theta)**6 + 3586297444752.04*cos(theta)**4 - 50158006220.3082*cos(theta)**2 + 112968482.478172)*sin(6*phi)

@torch.jit.script
def Yl30_m_minus_5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.20720631489845e-7*(1.0 - cos(theta)**2)**2.5*(1.88351561805034e+15*cos(theta)**25 - 9.57719805788307e+15*cos(theta)**23 + 2.1254658847758e+16*cos(theta)**21 - 2.70513839880557e+16*cos(theta)**19 + 2.181974840546e+16*cos(theta)**17 - 1.16371991495787e+16*cos(theta)**15 + 4.15614255342096e+15*cos(theta)**13 - 985346866768494.0*cos(theta)**11 + 150539104645187.0*cos(theta)**9 - 14003637641412.7*cos(theta)**7 + 717259488950.407*cos(theta)**5 - 16719335406.7694*cos(theta)**3 + 112968482.478172*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl30_m_minus_4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.64168346911826e-6*(1.0 - cos(theta)**2)**2*(72442908386551.5*cos(theta)**26 - 399049919078461.0*cos(theta)**24 + 966120856716275.0*cos(theta)**22 - 1.35256919940279e+15*cos(theta)**20 + 1.21220824474778e+15*cos(theta)**18 - 727324946848667.0*cos(theta)**16 + 296867325244354.0*cos(theta)**14 - 82112238897374.5*cos(theta)**12 + 15053910464518.7*cos(theta)**10 - 1750454705176.59*cos(theta)**8 + 119543248158.401*cos(theta)**6 - 4179833851.69235*cos(theta)**4 + 56484241.2390858*cos(theta)**2 - 124141.189536452)*sin(4*phi)

@torch.jit.script
def Yl30_m_minus_3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000110337600540934*(1.0 - cos(theta)**2)**1.5*(2683070680983.39*cos(theta)**27 - 15961996763138.5*cos(theta)**25 + 42005254639838.0*cos(theta)**23 - 64408057114418.3*cos(theta)**21 + 63800433934093.6*cos(theta)**19 - 42783820402862.8*cos(theta)**17 + 19791155016290.3*cos(theta)**15 - 6316326069028.81*cos(theta)**13 + 1368537314956.24*cos(theta)**11 - 194494967241.843*cos(theta)**9 + 17077606879.7716*cos(theta)**7 - 835966770.33847*cos(theta)**5 + 18828080.4130286*cos(theta)**3 - 124141.189536452*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl30_m_minus_2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00335397268176902*(1.0 - cos(theta)**2)*(95823952892.2638*cos(theta)**28 - 613922952428.402*cos(theta)**26 + 1750218943326.59*cos(theta)**24 - 2927638959746.29*cos(theta)**22 + 3190021696704.68*cos(theta)**20 - 2376878911270.15*cos(theta)**18 + 1236947188518.14*cos(theta)**16 - 451166147787.772*cos(theta)**14 + 114044776246.354*cos(theta)**12 - 19449496724.1843*cos(theta)**10 + 2134700859.97145*cos(theta)**8 - 139327795.056412*cos(theta)**6 + 4707020.10325715*cos(theta)**4 - 62070.5947682261*cos(theta)**2 + 134.351936727762)*sin(2*phi)

@torch.jit.script
def Yl30_m_minus_1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.102172379790475*(1.0 - cos(theta)**2)**0.5*(3304274237.66427*cos(theta)**29 - 22737887126.9779*cos(theta)**27 + 70008757733.0634*cos(theta)**25 - 127288650423.752*cos(theta)**23 + 151905795081.175*cos(theta)**21 - 125098890066.85*cos(theta)**19 + 72761599324.5966*cos(theta)**17 - 30077743185.8515*cos(theta)**15 + 8772675095.87335*cos(theta)**13 - 1768136065.83494*cos(theta)**11 + 237188984.441272*cos(theta)**9 - 19903970.7223445*cos(theta)**7 + 941404.02065143*cos(theta)**5 - 20690.1982560754*cos(theta)**3 + 134.351936727762*cos(theta))*sin(phi)

@torch.jit.script
def Yl30_m0(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 242669287.558974*cos(theta)**30 - 1789171865.90091*cos(theta)**28 + 5932517239.56617*cos(theta)**26 - 11685261229.4485*cos(theta)**24 + 15212887260.9801*cos(theta)**22 - 13781086107.0055*cos(theta)**20 + 8906144082.75869*cos(theta)**18 - 4141763053.68413*cos(theta)**16 + 1380587684.56138*cos(theta)**14 - 324634313.423993*cos(theta)**12 + 52258206.5511794*cos(theta)**10 - 5481630.05781602*cos(theta)**8 + 345688.382024433*cos(theta)**6 - 11396.3202865198*cos(theta)**4 + 148.004159565192*cos(theta)**2 - 0.318288515193961

@torch.jit.script
def Yl30_m1(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.102172379790475*(1.0 - cos(theta)**2)**0.5*(3304274237.66427*cos(theta)**29 - 22737887126.9779*cos(theta)**27 + 70008757733.0634*cos(theta)**25 - 127288650423.752*cos(theta)**23 + 151905795081.175*cos(theta)**21 - 125098890066.85*cos(theta)**19 + 72761599324.5966*cos(theta)**17 - 30077743185.8515*cos(theta)**15 + 8772675095.87335*cos(theta)**13 - 1768136065.83494*cos(theta)**11 + 237188984.441272*cos(theta)**9 - 19903970.7223445*cos(theta)**7 + 941404.02065143*cos(theta)**5 - 20690.1982560754*cos(theta)**3 + 134.351936727762*cos(theta))*cos(phi)

@torch.jit.script
def Yl30_m2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.00335397268176902*(1.0 - cos(theta)**2)*(95823952892.2638*cos(theta)**28 - 613922952428.402*cos(theta)**26 + 1750218943326.59*cos(theta)**24 - 2927638959746.29*cos(theta)**22 + 3190021696704.68*cos(theta)**20 - 2376878911270.15*cos(theta)**18 + 1236947188518.14*cos(theta)**16 - 451166147787.772*cos(theta)**14 + 114044776246.354*cos(theta)**12 - 19449496724.1843*cos(theta)**10 + 2134700859.97145*cos(theta)**8 - 139327795.056412*cos(theta)**6 + 4707020.10325715*cos(theta)**4 - 62070.5947682261*cos(theta)**2 + 134.351936727762)*cos(2*phi)

@torch.jit.script
def Yl30_m3(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.000110337600540934*(1.0 - cos(theta)**2)**1.5*(2683070680983.39*cos(theta)**27 - 15961996763138.5*cos(theta)**25 + 42005254639838.0*cos(theta)**23 - 64408057114418.3*cos(theta)**21 + 63800433934093.6*cos(theta)**19 - 42783820402862.8*cos(theta)**17 + 19791155016290.3*cos(theta)**15 - 6316326069028.81*cos(theta)**13 + 1368537314956.24*cos(theta)**11 - 194494967241.843*cos(theta)**9 + 17077606879.7716*cos(theta)**7 - 835966770.33847*cos(theta)**5 + 18828080.4130286*cos(theta)**3 - 124141.189536452*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl30_m4(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.64168346911826e-6*(1.0 - cos(theta)**2)**2*(72442908386551.5*cos(theta)**26 - 399049919078461.0*cos(theta)**24 + 966120856716275.0*cos(theta)**22 - 1.35256919940279e+15*cos(theta)**20 + 1.21220824474778e+15*cos(theta)**18 - 727324946848667.0*cos(theta)**16 + 296867325244354.0*cos(theta)**14 - 82112238897374.5*cos(theta)**12 + 15053910464518.7*cos(theta)**10 - 1750454705176.59*cos(theta)**8 + 119543248158.401*cos(theta)**6 - 4179833851.69235*cos(theta)**4 + 56484241.2390858*cos(theta)**2 - 124141.189536452)*cos(4*phi)

@torch.jit.script
def Yl30_m5(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.20720631489845e-7*(1.0 - cos(theta)**2)**2.5*(1.88351561805034e+15*cos(theta)**25 - 9.57719805788307e+15*cos(theta)**23 + 2.1254658847758e+16*cos(theta)**21 - 2.70513839880557e+16*cos(theta)**19 + 2.181974840546e+16*cos(theta)**17 - 1.16371991495787e+16*cos(theta)**15 + 4.15614255342096e+15*cos(theta)**13 - 985346866768494.0*cos(theta)**11 + 150539104645187.0*cos(theta)**9 - 14003637641412.7*cos(theta)**7 + 717259488950.407*cos(theta)**5 - 16719335406.7694*cos(theta)**3 + 112968482.478172*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl30_m6(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.02402104966148e-9*(1.0 - cos(theta)**2)**3*(4.70878904512584e+16*cos(theta)**24 - 2.20275555331311e+17*cos(theta)**22 + 4.46347835802919e+17*cos(theta)**20 - 5.13976295773058e+17*cos(theta)**18 + 3.7093572289282e+17*cos(theta)**16 - 1.7455798724368e+17*cos(theta)**14 + 5.40298531944724e+16*cos(theta)**12 - 1.08388155344534e+16*cos(theta)**10 + 1.35485194180668e+15*cos(theta)**8 - 98025463489889.0*cos(theta)**6 + 3586297444752.04*cos(theta)**4 - 50158006220.3082*cos(theta)**2 + 112968482.478172)*cos(6*phi)

@torch.jit.script
def Yl30_m7(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.3503730468945e-10*(1.0 - cos(theta)**2)**3.5*(1.1301093708302e+18*cos(theta)**23 - 4.84606221728884e+18*cos(theta)**21 + 8.92695671605838e+18*cos(theta)**19 - 9.25157332391505e+18*cos(theta)**17 + 5.93497156628513e+18*cos(theta)**15 - 2.44381182141152e+18*cos(theta)**13 + 6.48358238333669e+17*cos(theta)**11 - 1.08388155344534e+17*cos(theta)**9 + 1.08388155344534e+16*cos(theta)**7 - 588152780939334.0*cos(theta)**5 + 14345189779008.1*cos(theta)**3 - 100316012440.616*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl30_m8(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.56770496751288e-12*(1.0 - cos(theta)**2)**4*(2.59925155290947e+19*cos(theta)**22 - 1.01767306563066e+20*cos(theta)**20 + 1.69612177605109e+20*cos(theta)**18 - 1.57276746506556e+20*cos(theta)**16 + 8.90245734942769e+19*cos(theta)**14 - 3.17695536783498e+19*cos(theta)**12 + 7.13194062167036e+18*cos(theta)**10 - 9.75493398100809e+17*cos(theta)**8 + 7.58717087411741e+16*cos(theta)**6 - 2.94076390469667e+15*cos(theta)**4 + 43035569337024.4*cos(theta)**2 - 100316012440.616)*cos(8*phi)

@torch.jit.script
def Yl30_m9(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.55938876429517e-13*(1.0 - cos(theta)**2)**4.5*(5.71835341640083e+20*cos(theta)**21 - 2.03534613126131e+21*cos(theta)**19 + 3.05301919689197e+21*cos(theta)**17 - 2.51642794410489e+21*cos(theta)**15 + 1.24634402891988e+21*cos(theta)**13 - 3.81234644140197e+20*cos(theta)**11 + 7.13194062167036e+19*cos(theta)**9 - 7.80394718480647e+18*cos(theta)**7 + 4.55230252447044e+17*cos(theta)**5 - 1.17630556187867e+16*cos(theta)**3 + 86071138674048.8*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl30_m10(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.38040239932763e-15*(1.0 - cos(theta)**2)**5*(1.20085421744417e+22*cos(theta)**20 - 3.86715764939649e+22*cos(theta)**18 + 5.19013263471634e+22*cos(theta)**16 - 3.77464191615734e+22*cos(theta)**14 + 1.62024723759584e+22*cos(theta)**12 - 4.19358108554217e+21*cos(theta)**10 + 6.41874655950332e+20*cos(theta)**8 - 5.46276302936453e+19*cos(theta)**6 + 2.27615126223522e+18*cos(theta)**4 - 3.528916685636e+16*cos(theta)**2 + 86071138674048.8)*cos(10*phi)

@torch.jit.script
def Yl30_m11(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.87891801956087e-16*(1.0 - cos(theta)**2)**5.5*(2.40170843488835e+23*cos(theta)**19 - 6.96088376891368e+23*cos(theta)**17 + 8.30421221554615e+23*cos(theta)**15 - 5.28449868262028e+23*cos(theta)**13 + 1.94429668511501e+23*cos(theta)**11 - 4.19358108554217e+22*cos(theta)**9 + 5.13499724760266e+21*cos(theta)**7 - 3.27765781761872e+20*cos(theta)**5 + 9.10460504894089e+18*cos(theta)**3 - 7.05783337127201e+16*cos(theta))*cos(11*phi)

@torch.jit.script
def Yl30_m12(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.65129768956931e-18*(1.0 - cos(theta)**2)**6*(4.56324602628786e+24*cos(theta)**18 - 1.18335024071533e+25*cos(theta)**16 + 1.24563183233192e+25*cos(theta)**14 - 6.86984828740636e+24*cos(theta)**12 + 2.13872635362651e+24*cos(theta)**10 - 3.77422297698795e+23*cos(theta)**8 + 3.59449807332186e+22*cos(theta)**6 - 1.63882890880936e+21*cos(theta)**4 + 2.73138151468227e+19*cos(theta)**2 - 7.05783337127201e+16)*cos(12*phi)

@torch.jit.script
def Yl30_m13(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.39075958422627e-19*(1.0 - cos(theta)**2)**6.5*(8.21384284731815e+25*cos(theta)**17 - 1.89336038514452e+26*cos(theta)**15 + 1.74388456526469e+26*cos(theta)**13 - 8.24381794488763e+25*cos(theta)**11 + 2.13872635362651e+25*cos(theta)**9 - 3.01937838159036e+24*cos(theta)**7 + 2.15669884399312e+23*cos(theta)**5 - 6.55531563523744e+21*cos(theta)**3 + 5.46276302936453e+19*cos(theta))*cos(13*phi)

@torch.jit.script
def Yl30_m14(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 8.74148278331158e-21*(1.0 - cos(theta)**2)**7*(1.39635328404408e+27*cos(theta)**16 - 2.84004057771678e+27*cos(theta)**14 + 2.2670499348441e+27*cos(theta)**12 - 9.06819973937639e+26*cos(theta)**10 + 1.92485371826386e+26*cos(theta)**8 - 2.11356486711326e+25*cos(theta)**6 + 1.07834942199656e+24*cos(theta)**4 - 1.96659469057123e+22*cos(theta)**2 + 5.46276302936453e+19)*cos(14*phi)

@torch.jit.script
def Yl30_m15(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.25775828793813e-22*(1.0 - cos(theta)**2)**7.5*(2.23416525447054e+28*cos(theta)**15 - 3.9760568088035e+28*cos(theta)**13 + 2.72045992181292e+28*cos(theta)**11 - 9.06819973937639e+27*cos(theta)**9 + 1.53988297461109e+27*cos(theta)**7 - 1.26813892026795e+26*cos(theta)**5 + 4.31339768798623e+24*cos(theta)**3 - 3.93318938114246e+22*cos(theta))*cos(15*phi)

@torch.jit.script
def Yl30_m16(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.24020738463486e-23*(1.0 - cos(theta)**2)**8*(3.3512478817058e+29*cos(theta)**14 - 5.16887385144454e+29*cos(theta)**12 + 2.99250591399421e+29*cos(theta)**10 - 8.16137976543875e+28*cos(theta)**8 + 1.07791808222776e+28*cos(theta)**6 - 6.34069460133976e+26*cos(theta)**4 + 1.29401930639587e+25*cos(theta)**2 - 3.93318938114246e+22)*cos(16*phi)

@torch.jit.script
def Yl30_m17(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 4.83483175810931e-25*(1.0 - cos(theta)**2)**8.5*(4.69174703438813e+30*cos(theta)**13 - 6.20264862173345e+30*cos(theta)**11 + 2.99250591399421e+30*cos(theta)**9 - 6.529103812351e+29*cos(theta)**7 + 6.46750849336656e+28*cos(theta)**5 - 2.53627784053591e+27*cos(theta)**3 + 2.58803861279174e+25*cos(theta))*cos(17*phi)

@torch.jit.script
def Yl30_m18(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.93548170846062e-26*(1.0 - cos(theta)**2)**9*(6.09927114470456e+31*cos(theta)**12 - 6.8229134839068e+31*cos(theta)**10 + 2.69325532259479e+31*cos(theta)**8 - 4.5703726686457e+30*cos(theta)**6 + 3.23375424668328e+29*cos(theta)**4 - 7.60883352160772e+27*cos(theta)**2 + 2.58803861279174e+25)*cos(18*phi)

@torch.jit.script
def Yl30_m19(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.98179203850954e-28*(1.0 - cos(theta)**2)**9.5*(7.31912537364547e+32*cos(theta)**11 - 6.8229134839068e+32*cos(theta)**9 + 2.15460425807583e+32*cos(theta)**7 - 2.74222360118742e+31*cos(theta)**5 + 1.29350169867331e+30*cos(theta)**3 - 1.52176670432154e+28*cos(theta))*cos(19*phi)

@torch.jit.script
def Yl30_m20(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.40344756082348e-29*(1.0 - cos(theta)**2)**10*(8.05103791101002e+33*cos(theta)**10 - 6.14062213551612e+33*cos(theta)**8 + 1.50822298065308e+33*cos(theta)**6 - 1.37111180059371e+32*cos(theta)**4 + 3.88050509601994e+30*cos(theta)**2 - 1.52176670432154e+28)*cos(20*phi)

@torch.jit.script
def Yl30_m21(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.5070719110102e-30*(1.0 - cos(theta)**2)**10.5*(8.05103791101002e+34*cos(theta)**9 - 4.9124977084129e+34*cos(theta)**7 + 9.04933788391849e+33*cos(theta)**5 - 5.48444720237484e+32*cos(theta)**3 + 7.76101019203987e+30*cos(theta))*cos(21*phi)

@torch.jit.script
def Yl30_m22(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 6.96644237302409e-32*(1.0 - cos(theta)**2)**11*(7.24593411990902e+35*cos(theta)**8 - 3.43874839588903e+35*cos(theta)**6 + 4.52466894195925e+34*cos(theta)**4 - 1.64533416071245e+33*cos(theta)**2 + 7.76101019203987e+30)*cos(22*phi)

@torch.jit.script
def Yl30_m23(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.38320349392245e-33*(1.0 - cos(theta)**2)**11.5*(5.79674729592722e+36*cos(theta)**7 - 2.06324903753342e+36*cos(theta)**5 + 1.8098675767837e+35*cos(theta)**3 - 3.29066832142491e+33*cos(theta))*cos(23*phi)

@torch.jit.script
def Yl30_m24(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 1.74013210905229e-34*(1.0 - cos(theta)**2)**12*(4.05772310714905e+37*cos(theta)**6 - 1.03162451876671e+37*cos(theta)**4 + 5.42960273035109e+35*cos(theta)**2 - 3.29066832142491e+33)*cos(24*phi)

@torch.jit.script
def Yl30_m25(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 9.57911199299742e-36*(1.0 - cos(theta)**2)**12.5*(2.43463386428943e+38*cos(theta)**5 - 4.12649807506683e+37*cos(theta)**3 + 1.08592054607022e+36*cos(theta))*cos(25*phi)

@torch.jit.script
def Yl30_m26(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 5.72461435302436e-37*(1.0 - cos(theta)**2)**13*(1.21731693214472e+39*cos(theta)**4 - 1.23794942252005e+38*cos(theta)**2 + 1.08592054607022e+36)*cos(26*phi)

@torch.jit.script
def Yl30_m27(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 3.79121847114987e-38*(1.0 - cos(theta)**2)**13.5*(4.86926772857886e+39*cos(theta)**3 - 2.4758988450401e+38*cos(theta))*cos(27*phi)

@torch.jit.script
def Yl30_m28(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.87411530575892e-39*(1.0 - cos(theta)**2)**14*(1.46078031857366e+40*cos(theta)**2 - 2.4758988450401e+38)*cos(28*phi)

@torch.jit.script
def Yl30_m29(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 7.72997498267602*(1.0 - cos(theta)**2)**14.5*cos(29*phi)*cos(theta)

@torch.jit.script
def Yl30_m30(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.997935479150139*(1.0 - cos(theta)**2)**15*cos(30*phi)

_SH_DISPATCH = {
    (0,0): Yl0_m0,
    (1,-1): Yl1_m_minus_1,
    (1,0): Yl1_m0,
    (1,1): Yl1_m1,
    (2,-2): Yl2_m_minus_2,
    (2,-1): Yl2_m_minus_1,
    (2,0): Yl2_m0,
    (2,1): Yl2_m1,
    (2,2): Yl2_m2,
    (3,-3): Yl3_m_minus_3,
    (3,-2): Yl3_m_minus_2,
    (3,-1): Yl3_m_minus_1,
    (3,0): Yl3_m0,
    (3,1): Yl3_m1,
    (3,2): Yl3_m2,
    (3,3): Yl3_m3,
    (4,-4): Yl4_m_minus_4,
    (4,-3): Yl4_m_minus_3,
    (4,-2): Yl4_m_minus_2,
    (4,-1): Yl4_m_minus_1,
    (4,0): Yl4_m0,
    (4,1): Yl4_m1,
    (4,2): Yl4_m2,
    (4,3): Yl4_m3,
    (4,4): Yl4_m4,
    (5,-5): Yl5_m_minus_5,
    (5,-4): Yl5_m_minus_4,
    (5,-3): Yl5_m_minus_3,
    (5,-2): Yl5_m_minus_2,
    (5,-1): Yl5_m_minus_1,
    (5,0): Yl5_m0,
    (5,1): Yl5_m1,
    (5,2): Yl5_m2,
    (5,3): Yl5_m3,
    (5,4): Yl5_m4,
    (5,5): Yl5_m5,
    (6,-6): Yl6_m_minus_6,
    (6,-5): Yl6_m_minus_5,
    (6,-4): Yl6_m_minus_4,
    (6,-3): Yl6_m_minus_3,
    (6,-2): Yl6_m_minus_2,
    (6,-1): Yl6_m_minus_1,
    (6,0): Yl6_m0,
    (6,1): Yl6_m1,
    (6,2): Yl6_m2,
    (6,3): Yl6_m3,
    (6,4): Yl6_m4,
    (6,5): Yl6_m5,
    (6,6): Yl6_m6,
    (7,-7): Yl7_m_minus_7,
    (7,-6): Yl7_m_minus_6,
    (7,-5): Yl7_m_minus_5,
    (7,-4): Yl7_m_minus_4,
    (7,-3): Yl7_m_minus_3,
    (7,-2): Yl7_m_minus_2,
    (7,-1): Yl7_m_minus_1,
    (7,0): Yl7_m0,
    (7,1): Yl7_m1,
    (7,2): Yl7_m2,
    (7,3): Yl7_m3,
    (7,4): Yl7_m4,
    (7,5): Yl7_m5,
    (7,6): Yl7_m6,
    (7,7): Yl7_m7,
    (8,-8): Yl8_m_minus_8,
    (8,-7): Yl8_m_minus_7,
    (8,-6): Yl8_m_minus_6,
    (8,-5): Yl8_m_minus_5,
    (8,-4): Yl8_m_minus_4,
    (8,-3): Yl8_m_minus_3,
    (8,-2): Yl8_m_minus_2,
    (8,-1): Yl8_m_minus_1,
    (8,0): Yl8_m0,
    (8,1): Yl8_m1,
    (8,2): Yl8_m2,
    (8,3): Yl8_m3,
    (8,4): Yl8_m4,
    (8,5): Yl8_m5,
    (8,6): Yl8_m6,
    (8,7): Yl8_m7,
    (8,8): Yl8_m8,
    (9,-9): Yl9_m_minus_9,
    (9,-8): Yl9_m_minus_8,
    (9,-7): Yl9_m_minus_7,
    (9,-6): Yl9_m_minus_6,
    (9,-5): Yl9_m_minus_5,
    (9,-4): Yl9_m_minus_4,
    (9,-3): Yl9_m_minus_3,
    (9,-2): Yl9_m_minus_2,
    (9,-1): Yl9_m_minus_1,
    (9,0): Yl9_m0,
    (9,1): Yl9_m1,
    (9,2): Yl9_m2,
    (9,3): Yl9_m3,
    (9,4): Yl9_m4,
    (9,5): Yl9_m5,
    (9,6): Yl9_m6,
    (9,7): Yl9_m7,
    (9,8): Yl9_m8,
    (9,9): Yl9_m9,
    (10,-10): Yl10_m_minus_10,
    (10,-9): Yl10_m_minus_9,
    (10,-8): Yl10_m_minus_8,
    (10,-7): Yl10_m_minus_7,
    (10,-6): Yl10_m_minus_6,
    (10,-5): Yl10_m_minus_5,
    (10,-4): Yl10_m_minus_4,
    (10,-3): Yl10_m_minus_3,
    (10,-2): Yl10_m_minus_2,
    (10,-1): Yl10_m_minus_1,
    (10,0): Yl10_m0,
    (10,1): Yl10_m1,
    (10,2): Yl10_m2,
    (10,3): Yl10_m3,
    (10,4): Yl10_m4,
    (10,5): Yl10_m5,
    (10,6): Yl10_m6,
    (10,7): Yl10_m7,
    (10,8): Yl10_m8,
    (10,9): Yl10_m9,
    (10,10): Yl10_m10,
    (11,-11): Yl11_m_minus_11,
    (11,-10): Yl11_m_minus_10,
    (11,-9): Yl11_m_minus_9,
    (11,-8): Yl11_m_minus_8,
    (11,-7): Yl11_m_minus_7,
    (11,-6): Yl11_m_minus_6,
    (11,-5): Yl11_m_minus_5,
    (11,-4): Yl11_m_minus_4,
    (11,-3): Yl11_m_minus_3,
    (11,-2): Yl11_m_minus_2,
    (11,-1): Yl11_m_minus_1,
    (11,0): Yl11_m0,
    (11,1): Yl11_m1,
    (11,2): Yl11_m2,
    (11,3): Yl11_m3,
    (11,4): Yl11_m4,
    (11,5): Yl11_m5,
    (11,6): Yl11_m6,
    (11,7): Yl11_m7,
    (11,8): Yl11_m8,
    (11,9): Yl11_m9,
    (11,10): Yl11_m10,
    (11,11): Yl11_m11,
    (12,-12): Yl12_m_minus_12,
    (12,-11): Yl12_m_minus_11,
    (12,-10): Yl12_m_minus_10,
    (12,-9): Yl12_m_minus_9,
    (12,-8): Yl12_m_minus_8,
    (12,-7): Yl12_m_minus_7,
    (12,-6): Yl12_m_minus_6,
    (12,-5): Yl12_m_minus_5,
    (12,-4): Yl12_m_minus_4,
    (12,-3): Yl12_m_minus_3,
    (12,-2): Yl12_m_minus_2,
    (12,-1): Yl12_m_minus_1,
    (12,0): Yl12_m0,
    (12,1): Yl12_m1,
    (12,2): Yl12_m2,
    (12,3): Yl12_m3,
    (12,4): Yl12_m4,
    (12,5): Yl12_m5,
    (12,6): Yl12_m6,
    (12,7): Yl12_m7,
    (12,8): Yl12_m8,
    (12,9): Yl12_m9,
    (12,10): Yl12_m10,
    (12,11): Yl12_m11,
    (12,12): Yl12_m12,
    (13,-13): Yl13_m_minus_13,
    (13,-12): Yl13_m_minus_12,
    (13,-11): Yl13_m_minus_11,
    (13,-10): Yl13_m_minus_10,
    (13,-9): Yl13_m_minus_9,
    (13,-8): Yl13_m_minus_8,
    (13,-7): Yl13_m_minus_7,
    (13,-6): Yl13_m_minus_6,
    (13,-5): Yl13_m_minus_5,
    (13,-4): Yl13_m_minus_4,
    (13,-3): Yl13_m_minus_3,
    (13,-2): Yl13_m_minus_2,
    (13,-1): Yl13_m_minus_1,
    (13,0): Yl13_m0,
    (13,1): Yl13_m1,
    (13,2): Yl13_m2,
    (13,3): Yl13_m3,
    (13,4): Yl13_m4,
    (13,5): Yl13_m5,
    (13,6): Yl13_m6,
    (13,7): Yl13_m7,
    (13,8): Yl13_m8,
    (13,9): Yl13_m9,
    (13,10): Yl13_m10,
    (13,11): Yl13_m11,
    (13,12): Yl13_m12,
    (13,13): Yl13_m13,
    (14,-14): Yl14_m_minus_14,
    (14,-13): Yl14_m_minus_13,
    (14,-12): Yl14_m_minus_12,
    (14,-11): Yl14_m_minus_11,
    (14,-10): Yl14_m_minus_10,
    (14,-9): Yl14_m_minus_9,
    (14,-8): Yl14_m_minus_8,
    (14,-7): Yl14_m_minus_7,
    (14,-6): Yl14_m_minus_6,
    (14,-5): Yl14_m_minus_5,
    (14,-4): Yl14_m_minus_4,
    (14,-3): Yl14_m_minus_3,
    (14,-2): Yl14_m_minus_2,
    (14,-1): Yl14_m_minus_1,
    (14,0): Yl14_m0,
    (14,1): Yl14_m1,
    (14,2): Yl14_m2,
    (14,3): Yl14_m3,
    (14,4): Yl14_m4,
    (14,5): Yl14_m5,
    (14,6): Yl14_m6,
    (14,7): Yl14_m7,
    (14,8): Yl14_m8,
    (14,9): Yl14_m9,
    (14,10): Yl14_m10,
    (14,11): Yl14_m11,
    (14,12): Yl14_m12,
    (14,13): Yl14_m13,
    (14,14): Yl14_m14,
    (15,-15): Yl15_m_minus_15,
    (15,-14): Yl15_m_minus_14,
    (15,-13): Yl15_m_minus_13,
    (15,-12): Yl15_m_minus_12,
    (15,-11): Yl15_m_minus_11,
    (15,-10): Yl15_m_minus_10,
    (15,-9): Yl15_m_minus_9,
    (15,-8): Yl15_m_minus_8,
    (15,-7): Yl15_m_minus_7,
    (15,-6): Yl15_m_minus_6,
    (15,-5): Yl15_m_minus_5,
    (15,-4): Yl15_m_minus_4,
    (15,-3): Yl15_m_minus_3,
    (15,-2): Yl15_m_minus_2,
    (15,-1): Yl15_m_minus_1,
    (15,0): Yl15_m0,
    (15,1): Yl15_m1,
    (15,2): Yl15_m2,
    (15,3): Yl15_m3,
    (15,4): Yl15_m4,
    (15,5): Yl15_m5,
    (15,6): Yl15_m6,
    (15,7): Yl15_m7,
    (15,8): Yl15_m8,
    (15,9): Yl15_m9,
    (15,10): Yl15_m10,
    (15,11): Yl15_m11,
    (15,12): Yl15_m12,
    (15,13): Yl15_m13,
    (15,14): Yl15_m14,
    (15,15): Yl15_m15,
    (16,-16): Yl16_m_minus_16,
    (16,-15): Yl16_m_minus_15,
    (16,-14): Yl16_m_minus_14,
    (16,-13): Yl16_m_minus_13,
    (16,-12): Yl16_m_minus_12,
    (16,-11): Yl16_m_minus_11,
    (16,-10): Yl16_m_minus_10,
    (16,-9): Yl16_m_minus_9,
    (16,-8): Yl16_m_minus_8,
    (16,-7): Yl16_m_minus_7,
    (16,-6): Yl16_m_minus_6,
    (16,-5): Yl16_m_minus_5,
    (16,-4): Yl16_m_minus_4,
    (16,-3): Yl16_m_minus_3,
    (16,-2): Yl16_m_minus_2,
    (16,-1): Yl16_m_minus_1,
    (16,0): Yl16_m0,
    (16,1): Yl16_m1,
    (16,2): Yl16_m2,
    (16,3): Yl16_m3,
    (16,4): Yl16_m4,
    (16,5): Yl16_m5,
    (16,6): Yl16_m6,
    (16,7): Yl16_m7,
    (16,8): Yl16_m8,
    (16,9): Yl16_m9,
    (16,10): Yl16_m10,
    (16,11): Yl16_m11,
    (16,12): Yl16_m12,
    (16,13): Yl16_m13,
    (16,14): Yl16_m14,
    (16,15): Yl16_m15,
    (16,16): Yl16_m16,
    (17,-17): Yl17_m_minus_17,
    (17,-16): Yl17_m_minus_16,
    (17,-15): Yl17_m_minus_15,
    (17,-14): Yl17_m_minus_14,
    (17,-13): Yl17_m_minus_13,
    (17,-12): Yl17_m_minus_12,
    (17,-11): Yl17_m_minus_11,
    (17,-10): Yl17_m_minus_10,
    (17,-9): Yl17_m_minus_9,
    (17,-8): Yl17_m_minus_8,
    (17,-7): Yl17_m_minus_7,
    (17,-6): Yl17_m_minus_6,
    (17,-5): Yl17_m_minus_5,
    (17,-4): Yl17_m_minus_4,
    (17,-3): Yl17_m_minus_3,
    (17,-2): Yl17_m_minus_2,
    (17,-1): Yl17_m_minus_1,
    (17,0): Yl17_m0,
    (17,1): Yl17_m1,
    (17,2): Yl17_m2,
    (17,3): Yl17_m3,
    (17,4): Yl17_m4,
    (17,5): Yl17_m5,
    (17,6): Yl17_m6,
    (17,7): Yl17_m7,
    (17,8): Yl17_m8,
    (17,9): Yl17_m9,
    (17,10): Yl17_m10,
    (17,11): Yl17_m11,
    (17,12): Yl17_m12,
    (17,13): Yl17_m13,
    (17,14): Yl17_m14,
    (17,15): Yl17_m15,
    (17,16): Yl17_m16,
    (17,17): Yl17_m17,
    (18,-18): Yl18_m_minus_18,
    (18,-17): Yl18_m_minus_17,
    (18,-16): Yl18_m_minus_16,
    (18,-15): Yl18_m_minus_15,
    (18,-14): Yl18_m_minus_14,
    (18,-13): Yl18_m_minus_13,
    (18,-12): Yl18_m_minus_12,
    (18,-11): Yl18_m_minus_11,
    (18,-10): Yl18_m_minus_10,
    (18,-9): Yl18_m_minus_9,
    (18,-8): Yl18_m_minus_8,
    (18,-7): Yl18_m_minus_7,
    (18,-6): Yl18_m_minus_6,
    (18,-5): Yl18_m_minus_5,
    (18,-4): Yl18_m_minus_4,
    (18,-3): Yl18_m_minus_3,
    (18,-2): Yl18_m_minus_2,
    (18,-1): Yl18_m_minus_1,
    (18,0): Yl18_m0,
    (18,1): Yl18_m1,
    (18,2): Yl18_m2,
    (18,3): Yl18_m3,
    (18,4): Yl18_m4,
    (18,5): Yl18_m5,
    (18,6): Yl18_m6,
    (18,7): Yl18_m7,
    (18,8): Yl18_m8,
    (18,9): Yl18_m9,
    (18,10): Yl18_m10,
    (18,11): Yl18_m11,
    (18,12): Yl18_m12,
    (18,13): Yl18_m13,
    (18,14): Yl18_m14,
    (18,15): Yl18_m15,
    (18,16): Yl18_m16,
    (18,17): Yl18_m17,
    (18,18): Yl18_m18,
    (19,-19): Yl19_m_minus_19,
    (19,-18): Yl19_m_minus_18,
    (19,-17): Yl19_m_minus_17,
    (19,-16): Yl19_m_minus_16,
    (19,-15): Yl19_m_minus_15,
    (19,-14): Yl19_m_minus_14,
    (19,-13): Yl19_m_minus_13,
    (19,-12): Yl19_m_minus_12,
    (19,-11): Yl19_m_minus_11,
    (19,-10): Yl19_m_minus_10,
    (19,-9): Yl19_m_minus_9,
    (19,-8): Yl19_m_minus_8,
    (19,-7): Yl19_m_minus_7,
    (19,-6): Yl19_m_minus_6,
    (19,-5): Yl19_m_minus_5,
    (19,-4): Yl19_m_minus_4,
    (19,-3): Yl19_m_minus_3,
    (19,-2): Yl19_m_minus_2,
    (19,-1): Yl19_m_minus_1,
    (19,0): Yl19_m0,
    (19,1): Yl19_m1,
    (19,2): Yl19_m2,
    (19,3): Yl19_m3,
    (19,4): Yl19_m4,
    (19,5): Yl19_m5,
    (19,6): Yl19_m6,
    (19,7): Yl19_m7,
    (19,8): Yl19_m8,
    (19,9): Yl19_m9,
    (19,10): Yl19_m10,
    (19,11): Yl19_m11,
    (19,12): Yl19_m12,
    (19,13): Yl19_m13,
    (19,14): Yl19_m14,
    (19,15): Yl19_m15,
    (19,16): Yl19_m16,
    (19,17): Yl19_m17,
    (19,18): Yl19_m18,
    (19,19): Yl19_m19,
    (20,-20): Yl20_m_minus_20,
    (20,-19): Yl20_m_minus_19,
    (20,-18): Yl20_m_minus_18,
    (20,-17): Yl20_m_minus_17,
    (20,-16): Yl20_m_minus_16,
    (20,-15): Yl20_m_minus_15,
    (20,-14): Yl20_m_minus_14,
    (20,-13): Yl20_m_minus_13,
    (20,-12): Yl20_m_minus_12,
    (20,-11): Yl20_m_minus_11,
    (20,-10): Yl20_m_minus_10,
    (20,-9): Yl20_m_minus_9,
    (20,-8): Yl20_m_minus_8,
    (20,-7): Yl20_m_minus_7,
    (20,-6): Yl20_m_minus_6,
    (20,-5): Yl20_m_minus_5,
    (20,-4): Yl20_m_minus_4,
    (20,-3): Yl20_m_minus_3,
    (20,-2): Yl20_m_minus_2,
    (20,-1): Yl20_m_minus_1,
    (20,0): Yl20_m0,
    (20,1): Yl20_m1,
    (20,2): Yl20_m2,
    (20,3): Yl20_m3,
    (20,4): Yl20_m4,
    (20,5): Yl20_m5,
    (20,6): Yl20_m6,
    (20,7): Yl20_m7,
    (20,8): Yl20_m8,
    (20,9): Yl20_m9,
    (20,10): Yl20_m10,
    (20,11): Yl20_m11,
    (20,12): Yl20_m12,
    (20,13): Yl20_m13,
    (20,14): Yl20_m14,
    (20,15): Yl20_m15,
    (20,16): Yl20_m16,
    (20,17): Yl20_m17,
    (20,18): Yl20_m18,
    (20,19): Yl20_m19,
    (20,20): Yl20_m20,
    (21,-21): Yl21_m_minus_21,
    (21,-20): Yl21_m_minus_20,
    (21,-19): Yl21_m_minus_19,
    (21,-18): Yl21_m_minus_18,
    (21,-17): Yl21_m_minus_17,
    (21,-16): Yl21_m_minus_16,
    (21,-15): Yl21_m_minus_15,
    (21,-14): Yl21_m_minus_14,
    (21,-13): Yl21_m_minus_13,
    (21,-12): Yl21_m_minus_12,
    (21,-11): Yl21_m_minus_11,
    (21,-10): Yl21_m_minus_10,
    (21,-9): Yl21_m_minus_9,
    (21,-8): Yl21_m_minus_8,
    (21,-7): Yl21_m_minus_7,
    (21,-6): Yl21_m_minus_6,
    (21,-5): Yl21_m_minus_5,
    (21,-4): Yl21_m_minus_4,
    (21,-3): Yl21_m_minus_3,
    (21,-2): Yl21_m_minus_2,
    (21,-1): Yl21_m_minus_1,
    (21,0): Yl21_m0,
    (21,1): Yl21_m1,
    (21,2): Yl21_m2,
    (21,3): Yl21_m3,
    (21,4): Yl21_m4,
    (21,5): Yl21_m5,
    (21,6): Yl21_m6,
    (21,7): Yl21_m7,
    (21,8): Yl21_m8,
    (21,9): Yl21_m9,
    (21,10): Yl21_m10,
    (21,11): Yl21_m11,
    (21,12): Yl21_m12,
    (21,13): Yl21_m13,
    (21,14): Yl21_m14,
    (21,15): Yl21_m15,
    (21,16): Yl21_m16,
    (21,17): Yl21_m17,
    (21,18): Yl21_m18,
    (21,19): Yl21_m19,
    (21,20): Yl21_m20,
    (21,21): Yl21_m21,
    (22,-22): Yl22_m_minus_22,
    (22,-21): Yl22_m_minus_21,
    (22,-20): Yl22_m_minus_20,
    (22,-19): Yl22_m_minus_19,
    (22,-18): Yl22_m_minus_18,
    (22,-17): Yl22_m_minus_17,
    (22,-16): Yl22_m_minus_16,
    (22,-15): Yl22_m_minus_15,
    (22,-14): Yl22_m_minus_14,
    (22,-13): Yl22_m_minus_13,
    (22,-12): Yl22_m_minus_12,
    (22,-11): Yl22_m_minus_11,
    (22,-10): Yl22_m_minus_10,
    (22,-9): Yl22_m_minus_9,
    (22,-8): Yl22_m_minus_8,
    (22,-7): Yl22_m_minus_7,
    (22,-6): Yl22_m_minus_6,
    (22,-5): Yl22_m_minus_5,
    (22,-4): Yl22_m_minus_4,
    (22,-3): Yl22_m_minus_3,
    (22,-2): Yl22_m_minus_2,
    (22,-1): Yl22_m_minus_1,
    (22,0): Yl22_m0,
    (22,1): Yl22_m1,
    (22,2): Yl22_m2,
    (22,3): Yl22_m3,
    (22,4): Yl22_m4,
    (22,5): Yl22_m5,
    (22,6): Yl22_m6,
    (22,7): Yl22_m7,
    (22,8): Yl22_m8,
    (22,9): Yl22_m9,
    (22,10): Yl22_m10,
    (22,11): Yl22_m11,
    (22,12): Yl22_m12,
    (22,13): Yl22_m13,
    (22,14): Yl22_m14,
    (22,15): Yl22_m15,
    (22,16): Yl22_m16,
    (22,17): Yl22_m17,
    (22,18): Yl22_m18,
    (22,19): Yl22_m19,
    (22,20): Yl22_m20,
    (22,21): Yl22_m21,
    (22,22): Yl22_m22,
    (23,-23): Yl23_m_minus_23,
    (23,-22): Yl23_m_minus_22,
    (23,-21): Yl23_m_minus_21,
    (23,-20): Yl23_m_minus_20,
    (23,-19): Yl23_m_minus_19,
    (23,-18): Yl23_m_minus_18,
    (23,-17): Yl23_m_minus_17,
    (23,-16): Yl23_m_minus_16,
    (23,-15): Yl23_m_minus_15,
    (23,-14): Yl23_m_minus_14,
    (23,-13): Yl23_m_minus_13,
    (23,-12): Yl23_m_minus_12,
    (23,-11): Yl23_m_minus_11,
    (23,-10): Yl23_m_minus_10,
    (23,-9): Yl23_m_minus_9,
    (23,-8): Yl23_m_minus_8,
    (23,-7): Yl23_m_minus_7,
    (23,-6): Yl23_m_minus_6,
    (23,-5): Yl23_m_minus_5,
    (23,-4): Yl23_m_minus_4,
    (23,-3): Yl23_m_minus_3,
    (23,-2): Yl23_m_minus_2,
    (23,-1): Yl23_m_minus_1,
    (23,0): Yl23_m0,
    (23,1): Yl23_m1,
    (23,2): Yl23_m2,
    (23,3): Yl23_m3,
    (23,4): Yl23_m4,
    (23,5): Yl23_m5,
    (23,6): Yl23_m6,
    (23,7): Yl23_m7,
    (23,8): Yl23_m8,
    (23,9): Yl23_m9,
    (23,10): Yl23_m10,
    (23,11): Yl23_m11,
    (23,12): Yl23_m12,
    (23,13): Yl23_m13,
    (23,14): Yl23_m14,
    (23,15): Yl23_m15,
    (23,16): Yl23_m16,
    (23,17): Yl23_m17,
    (23,18): Yl23_m18,
    (23,19): Yl23_m19,
    (23,20): Yl23_m20,
    (23,21): Yl23_m21,
    (23,22): Yl23_m22,
    (23,23): Yl23_m23,
    (24,-24): Yl24_m_minus_24,
    (24,-23): Yl24_m_minus_23,
    (24,-22): Yl24_m_minus_22,
    (24,-21): Yl24_m_minus_21,
    (24,-20): Yl24_m_minus_20,
    (24,-19): Yl24_m_minus_19,
    (24,-18): Yl24_m_minus_18,
    (24,-17): Yl24_m_minus_17,
    (24,-16): Yl24_m_minus_16,
    (24,-15): Yl24_m_minus_15,
    (24,-14): Yl24_m_minus_14,
    (24,-13): Yl24_m_minus_13,
    (24,-12): Yl24_m_minus_12,
    (24,-11): Yl24_m_minus_11,
    (24,-10): Yl24_m_minus_10,
    (24,-9): Yl24_m_minus_9,
    (24,-8): Yl24_m_minus_8,
    (24,-7): Yl24_m_minus_7,
    (24,-6): Yl24_m_minus_6,
    (24,-5): Yl24_m_minus_5,
    (24,-4): Yl24_m_minus_4,
    (24,-3): Yl24_m_minus_3,
    (24,-2): Yl24_m_minus_2,
    (24,-1): Yl24_m_minus_1,
    (24,0): Yl24_m0,
    (24,1): Yl24_m1,
    (24,2): Yl24_m2,
    (24,3): Yl24_m3,
    (24,4): Yl24_m4,
    (24,5): Yl24_m5,
    (24,6): Yl24_m6,
    (24,7): Yl24_m7,
    (24,8): Yl24_m8,
    (24,9): Yl24_m9,
    (24,10): Yl24_m10,
    (24,11): Yl24_m11,
    (24,12): Yl24_m12,
    (24,13): Yl24_m13,
    (24,14): Yl24_m14,
    (24,15): Yl24_m15,
    (24,16): Yl24_m16,
    (24,17): Yl24_m17,
    (24,18): Yl24_m18,
    (24,19): Yl24_m19,
    (24,20): Yl24_m20,
    (24,21): Yl24_m21,
    (24,22): Yl24_m22,
    (24,23): Yl24_m23,
    (24,24): Yl24_m24,
    (25,-25): Yl25_m_minus_25,
    (25,-24): Yl25_m_minus_24,
    (25,-23): Yl25_m_minus_23,
    (25,-22): Yl25_m_minus_22,
    (25,-21): Yl25_m_minus_21,
    (25,-20): Yl25_m_minus_20,
    (25,-19): Yl25_m_minus_19,
    (25,-18): Yl25_m_minus_18,
    (25,-17): Yl25_m_minus_17,
    (25,-16): Yl25_m_minus_16,
    (25,-15): Yl25_m_minus_15,
    (25,-14): Yl25_m_minus_14,
    (25,-13): Yl25_m_minus_13,
    (25,-12): Yl25_m_minus_12,
    (25,-11): Yl25_m_minus_11,
    (25,-10): Yl25_m_minus_10,
    (25,-9): Yl25_m_minus_9,
    (25,-8): Yl25_m_minus_8,
    (25,-7): Yl25_m_minus_7,
    (25,-6): Yl25_m_minus_6,
    (25,-5): Yl25_m_minus_5,
    (25,-4): Yl25_m_minus_4,
    (25,-3): Yl25_m_minus_3,
    (25,-2): Yl25_m_minus_2,
    (25,-1): Yl25_m_minus_1,
    (25,0): Yl25_m0,
    (25,1): Yl25_m1,
    (25,2): Yl25_m2,
    (25,3): Yl25_m3,
    (25,4): Yl25_m4,
    (25,5): Yl25_m5,
    (25,6): Yl25_m6,
    (25,7): Yl25_m7,
    (25,8): Yl25_m8,
    (25,9): Yl25_m9,
    (25,10): Yl25_m10,
    (25,11): Yl25_m11,
    (25,12): Yl25_m12,
    (25,13): Yl25_m13,
    (25,14): Yl25_m14,
    (25,15): Yl25_m15,
    (25,16): Yl25_m16,
    (25,17): Yl25_m17,
    (25,18): Yl25_m18,
    (25,19): Yl25_m19,
    (25,20): Yl25_m20,
    (25,21): Yl25_m21,
    (25,22): Yl25_m22,
    (25,23): Yl25_m23,
    (25,24): Yl25_m24,
    (25,25): Yl25_m25,
    (26,-26): Yl26_m_minus_26,
    (26,-25): Yl26_m_minus_25,
    (26,-24): Yl26_m_minus_24,
    (26,-23): Yl26_m_minus_23,
    (26,-22): Yl26_m_minus_22,
    (26,-21): Yl26_m_minus_21,
    (26,-20): Yl26_m_minus_20,
    (26,-19): Yl26_m_minus_19,
    (26,-18): Yl26_m_minus_18,
    (26,-17): Yl26_m_minus_17,
    (26,-16): Yl26_m_minus_16,
    (26,-15): Yl26_m_minus_15,
    (26,-14): Yl26_m_minus_14,
    (26,-13): Yl26_m_minus_13,
    (26,-12): Yl26_m_minus_12,
    (26,-11): Yl26_m_minus_11,
    (26,-10): Yl26_m_minus_10,
    (26,-9): Yl26_m_minus_9,
    (26,-8): Yl26_m_minus_8,
    (26,-7): Yl26_m_minus_7,
    (26,-6): Yl26_m_minus_6,
    (26,-5): Yl26_m_minus_5,
    (26,-4): Yl26_m_minus_4,
    (26,-3): Yl26_m_minus_3,
    (26,-2): Yl26_m_minus_2,
    (26,-1): Yl26_m_minus_1,
    (26,0): Yl26_m0,
    (26,1): Yl26_m1,
    (26,2): Yl26_m2,
    (26,3): Yl26_m3,
    (26,4): Yl26_m4,
    (26,5): Yl26_m5,
    (26,6): Yl26_m6,
    (26,7): Yl26_m7,
    (26,8): Yl26_m8,
    (26,9): Yl26_m9,
    (26,10): Yl26_m10,
    (26,11): Yl26_m11,
    (26,12): Yl26_m12,
    (26,13): Yl26_m13,
    (26,14): Yl26_m14,
    (26,15): Yl26_m15,
    (26,16): Yl26_m16,
    (26,17): Yl26_m17,
    (26,18): Yl26_m18,
    (26,19): Yl26_m19,
    (26,20): Yl26_m20,
    (26,21): Yl26_m21,
    (26,22): Yl26_m22,
    (26,23): Yl26_m23,
    (26,24): Yl26_m24,
    (26,25): Yl26_m25,
    (26,26): Yl26_m26,
    (27,-27): Yl27_m_minus_27,
    (27,-26): Yl27_m_minus_26,
    (27,-25): Yl27_m_minus_25,
    (27,-24): Yl27_m_minus_24,
    (27,-23): Yl27_m_minus_23,
    (27,-22): Yl27_m_minus_22,
    (27,-21): Yl27_m_minus_21,
    (27,-20): Yl27_m_minus_20,
    (27,-19): Yl27_m_minus_19,
    (27,-18): Yl27_m_minus_18,
    (27,-17): Yl27_m_minus_17,
    (27,-16): Yl27_m_minus_16,
    (27,-15): Yl27_m_minus_15,
    (27,-14): Yl27_m_minus_14,
    (27,-13): Yl27_m_minus_13,
    (27,-12): Yl27_m_minus_12,
    (27,-11): Yl27_m_minus_11,
    (27,-10): Yl27_m_minus_10,
    (27,-9): Yl27_m_minus_9,
    (27,-8): Yl27_m_minus_8,
    (27,-7): Yl27_m_minus_7,
    (27,-6): Yl27_m_minus_6,
    (27,-5): Yl27_m_minus_5,
    (27,-4): Yl27_m_minus_4,
    (27,-3): Yl27_m_minus_3,
    (27,-2): Yl27_m_minus_2,
    (27,-1): Yl27_m_minus_1,
    (27,0): Yl27_m0,
    (27,1): Yl27_m1,
    (27,2): Yl27_m2,
    (27,3): Yl27_m3,
    (27,4): Yl27_m4,
    (27,5): Yl27_m5,
    (27,6): Yl27_m6,
    (27,7): Yl27_m7,
    (27,8): Yl27_m8,
    (27,9): Yl27_m9,
    (27,10): Yl27_m10,
    (27,11): Yl27_m11,
    (27,12): Yl27_m12,
    (27,13): Yl27_m13,
    (27,14): Yl27_m14,
    (27,15): Yl27_m15,
    (27,16): Yl27_m16,
    (27,17): Yl27_m17,
    (27,18): Yl27_m18,
    (27,19): Yl27_m19,
    (27,20): Yl27_m20,
    (27,21): Yl27_m21,
    (27,22): Yl27_m22,
    (27,23): Yl27_m23,
    (27,24): Yl27_m24,
    (27,25): Yl27_m25,
    (27,26): Yl27_m26,
    (27,27): Yl27_m27,
    (28,-28): Yl28_m_minus_28,
    (28,-27): Yl28_m_minus_27,
    (28,-26): Yl28_m_minus_26,
    (28,-25): Yl28_m_minus_25,
    (28,-24): Yl28_m_minus_24,
    (28,-23): Yl28_m_minus_23,
    (28,-22): Yl28_m_minus_22,
    (28,-21): Yl28_m_minus_21,
    (28,-20): Yl28_m_minus_20,
    (28,-19): Yl28_m_minus_19,
    (28,-18): Yl28_m_minus_18,
    (28,-17): Yl28_m_minus_17,
    (28,-16): Yl28_m_minus_16,
    (28,-15): Yl28_m_minus_15,
    (28,-14): Yl28_m_minus_14,
    (28,-13): Yl28_m_minus_13,
    (28,-12): Yl28_m_minus_12,
    (28,-11): Yl28_m_minus_11,
    (28,-10): Yl28_m_minus_10,
    (28,-9): Yl28_m_minus_9,
    (28,-8): Yl28_m_minus_8,
    (28,-7): Yl28_m_minus_7,
    (28,-6): Yl28_m_minus_6,
    (28,-5): Yl28_m_minus_5,
    (28,-4): Yl28_m_minus_4,
    (28,-3): Yl28_m_minus_3,
    (28,-2): Yl28_m_minus_2,
    (28,-1): Yl28_m_minus_1,
    (28,0): Yl28_m0,
    (28,1): Yl28_m1,
    (28,2): Yl28_m2,
    (28,3): Yl28_m3,
    (28,4): Yl28_m4,
    (28,5): Yl28_m5,
    (28,6): Yl28_m6,
    (28,7): Yl28_m7,
    (28,8): Yl28_m8,
    (28,9): Yl28_m9,
    (28,10): Yl28_m10,
    (28,11): Yl28_m11,
    (28,12): Yl28_m12,
    (28,13): Yl28_m13,
    (28,14): Yl28_m14,
    (28,15): Yl28_m15,
    (28,16): Yl28_m16,
    (28,17): Yl28_m17,
    (28,18): Yl28_m18,
    (28,19): Yl28_m19,
    (28,20): Yl28_m20,
    (28,21): Yl28_m21,
    (28,22): Yl28_m22,
    (28,23): Yl28_m23,
    (28,24): Yl28_m24,
    (28,25): Yl28_m25,
    (28,26): Yl28_m26,
    (28,27): Yl28_m27,
    (28,28): Yl28_m28,
    (29,-29): Yl29_m_minus_29,
    (29,-28): Yl29_m_minus_28,
    (29,-27): Yl29_m_minus_27,
    (29,-26): Yl29_m_minus_26,
    (29,-25): Yl29_m_minus_25,
    (29,-24): Yl29_m_minus_24,
    (29,-23): Yl29_m_minus_23,
    (29,-22): Yl29_m_minus_22,
    (29,-21): Yl29_m_minus_21,
    (29,-20): Yl29_m_minus_20,
    (29,-19): Yl29_m_minus_19,
    (29,-18): Yl29_m_minus_18,
    (29,-17): Yl29_m_minus_17,
    (29,-16): Yl29_m_minus_16,
    (29,-15): Yl29_m_minus_15,
    (29,-14): Yl29_m_minus_14,
    (29,-13): Yl29_m_minus_13,
    (29,-12): Yl29_m_minus_12,
    (29,-11): Yl29_m_minus_11,
    (29,-10): Yl29_m_minus_10,
    (29,-9): Yl29_m_minus_9,
    (29,-8): Yl29_m_minus_8,
    (29,-7): Yl29_m_minus_7,
    (29,-6): Yl29_m_minus_6,
    (29,-5): Yl29_m_minus_5,
    (29,-4): Yl29_m_minus_4,
    (29,-3): Yl29_m_minus_3,
    (29,-2): Yl29_m_minus_2,
    (29,-1): Yl29_m_minus_1,
    (29,0): Yl29_m0,
    (29,1): Yl29_m1,
    (29,2): Yl29_m2,
    (29,3): Yl29_m3,
    (29,4): Yl29_m4,
    (29,5): Yl29_m5,
    (29,6): Yl29_m6,
    (29,7): Yl29_m7,
    (29,8): Yl29_m8,
    (29,9): Yl29_m9,
    (29,10): Yl29_m10,
    (29,11): Yl29_m11,
    (29,12): Yl29_m12,
    (29,13): Yl29_m13,
    (29,14): Yl29_m14,
    (29,15): Yl29_m15,
    (29,16): Yl29_m16,
    (29,17): Yl29_m17,
    (29,18): Yl29_m18,
    (29,19): Yl29_m19,
    (29,20): Yl29_m20,
    (29,21): Yl29_m21,
    (29,22): Yl29_m22,
    (29,23): Yl29_m23,
    (29,24): Yl29_m24,
    (29,25): Yl29_m25,
    (29,26): Yl29_m26,
    (29,27): Yl29_m27,
    (29,28): Yl29_m28,
    (29,29): Yl29_m29,
    (30,-30): Yl30_m_minus_30,
    (30,-29): Yl30_m_minus_29,
    (30,-28): Yl30_m_minus_28,
    (30,-27): Yl30_m_minus_27,
    (30,-26): Yl30_m_minus_26,
    (30,-25): Yl30_m_minus_25,
    (30,-24): Yl30_m_minus_24,
    (30,-23): Yl30_m_minus_23,
    (30,-22): Yl30_m_minus_22,
    (30,-21): Yl30_m_minus_21,
    (30,-20): Yl30_m_minus_20,
    (30,-19): Yl30_m_minus_19,
    (30,-18): Yl30_m_minus_18,
    (30,-17): Yl30_m_minus_17,
    (30,-16): Yl30_m_minus_16,
    (30,-15): Yl30_m_minus_15,
    (30,-14): Yl30_m_minus_14,
    (30,-13): Yl30_m_minus_13,
    (30,-12): Yl30_m_minus_12,
    (30,-11): Yl30_m_minus_11,
    (30,-10): Yl30_m_minus_10,
    (30,-9): Yl30_m_minus_9,
    (30,-8): Yl30_m_minus_8,
    (30,-7): Yl30_m_minus_7,
    (30,-6): Yl30_m_minus_6,
    (30,-5): Yl30_m_minus_5,
    (30,-4): Yl30_m_minus_4,
    (30,-3): Yl30_m_minus_3,
    (30,-2): Yl30_m_minus_2,
    (30,-1): Yl30_m_minus_1,
    (30,0): Yl30_m0,
    (30,1): Yl30_m1,
    (30,2): Yl30_m2,
    (30,3): Yl30_m3,
    (30,4): Yl30_m4,
    (30,5): Yl30_m5,
    (30,6): Yl30_m6,
    (30,7): Yl30_m7,
    (30,8): Yl30_m8,
    (30,9): Yl30_m9,
    (30,10): Yl30_m10,
    (30,11): Yl30_m11,
    (30,12): Yl30_m12,
    (30,13): Yl30_m13,
    (30,14): Yl30_m14,
    (30,15): Yl30_m15,
    (30,16): Yl30_m16,
    (30,17): Yl30_m17,
    (30,18): Yl30_m18,
    (30,19): Yl30_m19,
    (30,20): Yl30_m20,
    (30,21): Yl30_m21,
    (30,22): Yl30_m22,
    (30,23): Yl30_m23,
    (30,24): Yl30_m24,
    (30,25): Yl30_m25,
    (30,26): Yl30_m26,
    (30,27): Yl30_m27,
    (30,28): Yl30_m28,
    (30,29): Yl30_m29,
    (30,30): Yl30_m30,
}

