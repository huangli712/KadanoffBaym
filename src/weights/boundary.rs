// Project : KadanoffBaym
// File    : src/weights/boundary.rs
// Purpose : boundary convolution weight calculators: the analytic
//           R integral and the rcorr-style tables :

use crate::weights::poly;
use crate::weights::rational::Rational;

///
/// *Remarks* : ``\mathbf{R}`` *Integral*
///
/// In order to evaluate the following integration analytically
///
/// ```math
/// \begin{equation}
/// \mathbf{R} = \int^{m}_0 dx\ (m-x)^a x^b,
/// \end{equation}
/// ```
///
/// we execute the following python codes:
///
/// ```text
/// import sympy
/// x, a, b, m = sympy.symbols('x a b m')
/// c = sympy.integrate((m-x)**a*x**b,(x,0,m))
/// c = sympy.simplify(c)
/// sympy.pprint(c)
/// ```
///
/// The output is as follows:
///
/// ```math
/// \begin{equation}
/// \mathbf{R} = \frac{m^{a+b+1}\Gamma(a+1)\Gamma(b+1)}{\Gamma(a+b+2)},
/// \end{equation}
/// ```
///
/// where ``\Gamma`` is a special function.
///
/// *Remarks* : ``\Gamma`` *Function*
///
/// The definition for the ``\Gamma`` function is as follows:
///
/// ```math
/// \begin{equation}
/// \Gamma(z) = \int^{\infty}_0 e^{-t} t^{z - 1} dt,
/// \end{equation}
/// ```
///
/// where ``z = x + iy`` and 𝑥 > 0.
///
///
/// Boundary convolution weights `Wc[m-1][i][j]` for `m = 1..k-1`,
/// row-major `(k-1) * (k+1) * (k+1)`; empty for `k = 1`:
/// `Wc[m][i][j] = sum_{a,b} Wi[a][i] Wi[b][j] R(m, a, b)` with the analytic
/// integral
///
/// ```math
/// R(m, a, b) = \int^{m}_0 dx\ (m-x)^a x^b
///            = \frac{m^{a+b+1}\Gamma(a+1)\Gamma(b+1)}{\Gamma(a+b+2)}
///            = \frac{m^{a+b+1} a!\, b!}{(a+b+1)!},
/// ```
///
/// where the Gamma functions of the analytic result are replaced by
/// factorials: `a` and `b` are non-negative integers (the loop indices), so
/// `Gamma(a+1) = a!`, `Gamma(b+1) = b!`, and `Gamma(a+b+2) = (a+b+1)!`.
/// The sum is exact: every term is reduced, the numerators are accumulated
/// over the least common multiple of the denominators (the largest
/// intermediate integer is `1.7e23` at `k = 10`, well within `i128`), and
/// the fraction is converted with a single rounding.
pub(super) fn calc_boundary_convolution(k: i64) -> Vec<f64> {
    let n = (k + 1) as usize;
    let wi = poly::vandermonde_inverse(n);
    let m_dim = (k - 1) as usize;
    let mut wc = vec![0.0f64; m_dim * n * n];
    let fact = |x: usize| (1..=x as i128).product::<i128>();
    for m in 1..=(k - 1) {
        for i in 0..n {
            for j in 0..n {
                // the exact sum, kept as a fraction with a common denominator
                let (mut num, mut den) = (0i128, 1i128);
                for a in 0..n {
                    for b in 0..n {
                        let r = Rational::new(
                            (m as i128).pow((a + b + 1) as u32) * fact(a) * fact(b),
                            fact(a + b + 1),
                        );
                        let (x, y) = (&wi[a * n + i], &wi[b * n + j]);
                        let t = Rational::new(x.num * y.num * r.num, x.den * y.den * r.den);
                        // t.den > 0: bring both fractions to the least common denominator
                        let g = Rational::gcd(den, t.den);
                        num = num * (t.den / g) + t.num * (den / g);
                        den = den / g * t.den;
                    }
                }
                wc[((m as usize - 1) * n + i) * n + j] = ratio_to_f64(num, den);
            }
        }
    }
    wc
}

/// Number of bits of a non-zero number.
fn bit_length(x: u128) -> i32 {
    (128 - x.leading_zeros()) as i32
}

/// `floor(a * 2^s / b)` and the remainder `r < b` of that division, for
/// `s >= 0`: `a * 2^s / b = q + r / b`. The integer part is shifted, the
/// `s` fractional bits come from a bit-by-bit division of the remainder.
fn scaled_quotient(a: u128, b: u128, s: u32) -> (u128, u128) {
    let (mut q, mut r) = (a / b, a % b);
    for _ in 0..s {
        let (r2, over) = r.overflowing_shl(1);
        q <<= 1;
        if over || r2 >= b {
            // 2r - b lies in [0, b): the wrapped subtraction is exact
            r = r2.wrapping_sub(b);
            q |= 1;
        } else {
            r = r2;
        }
    }
    (q, r)
}

/// `num / den` rounded to the nearest `f64` (ties to even) with a single
/// rounding. `num as f64 / den as f64` rounds three times when an operand
/// exceeds `2^53` and misses the correctly rounded value for 26 of the 1089
/// fractions of the `k = 10` table. The value of `i128` operands lies in
/// `[2^-127, 2^127]`, so the result is always a finite normal number.
fn ratio_to_f64(num: i128, den: i128) -> f64 {
    assert!(den != 0, "ratio_to_f64: zero denominator");
    if num == 0 {
        return 0.0;
    }
    let (neg, a, b) = if den < 0 {
        (num > 0, num.unsigned_abs(), den.unsigned_abs())
    } else {
        (num < 0, num.unsigned_abs(), den.unsigned_abs())
    };
    // binary exponent e of a / b, 2^e <= a / b < 2^(e + 1): e0 or e0 - 1
    let e0 = bit_length(a) - bit_length(b);
    let e = if e0 >= 0 {
        // a / b >= 2^e0  <=>  floor(a / b) >= 2^e0
        if a / b >= 1u128 << e0 { e0 } else { e0 - 1 }
    } else {
        // a / b >= 2^e0  <=>  b / a <= 2^(-e0)
        let (q, r) = (b / a, b % a);
        if q < 1u128 << -e0 || (q == 1u128 << -e0 && r == 0) {
            e0
        } else {
            e0 - 1
        }
    };
    // a / b * 2^(53 - e) = q + rho with q in [2^53, 2^54) and a remainder
    // rho in [0, 1): the 53 significant bits are q >> 1, rounded once with
    // the guard bit q & 1 and the sticky remainder
    let (q, r) = if e <= 53 {
        scaled_quotient(a, b, (53 - e) as u32)
    } else {
        // a >= b * 2^e makes b * 2^(e - 53) smaller than 2^75
        let d = b << (e - 53) as u32;
        (a / d, a % d)
    };
    let mut s = q >> 1;
    if q & 1 == 1 && (r > 0 || s & 1 == 1) {
        s += 1;
    }
    let (s, e) = if s == 1u128 << 53 {
        // the rounding carries into the next binade
        (1u128 << 52, e + 1)
    } else {
        (s, e)
    };
    let out = ((e + 1023) as u64) << 52 | (s - (1u128 << 52)) as u64;
    if neg {
        f64::from_bits(out | (1 << 63))
    } else {
        f64::from_bits(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::weights::gregory::binomial;

    /// `(numerator, denominator, expected bit pattern)`, the expected values
    /// from exact rational arithmetic: random operands beyond `2^53`, large
    /// ratios, exact ties (odd significands round up, even ones stay) and the
    /// ends of the `i128` range.
    const RATIO_RANDOM_CASES: &[(i128, i128, u64)] = &[
        (
            -69530459081355572352873675697652086724,
            -14201368466926901744945431749842901015,
            0x4013958b5807345d,
        ),
        (
            -80206582859898877780229547129816843514,
            -13757361932145034038392281295152069329,
            0x40175201ad465041,
        ),
        (
            13931036729159653571336075208686411158,
            -63073112265819869100936091866154355971,
            0xbfcc458247718503,
        ),
        (
            67998280735649895056946247540722164215,
            -79521886813540377781428673567514099257,
            0xbfeb5ce3582daacf,
        ),
        (
            44754439938004293024532207456490414313,
            66874032992920234214123759158195460545,
            0x3fe56a5f686fdef9,
        ),
        (
            70280857846823569108567320041501915591,
            43519090926992530840717053824783301288,
            0x3ff9d6cea14135cc,
        ),
        (
            -20044440361513756236811695088559407357,
            44816713311357056264431630412815310805,
            0xbfdc9fcdefeac7bc,
        ),
        (
            -57231960119992887367959019586569357480,
            26683407992849653415299443660366603059,
            0xc00128a825e88d6b,
        ),
        (
            7054715538709692299593264098841581783,
            71398988299642436792844544735131241880,
            0x3fb94b695f0f27ea,
        ),
        (
            -23608089227682863494079946231071936873,
            20801339906359177976823693577199522322,
            0xbff228ad989a17b8,
        ),
        (
            -9499515180615058508719500935131600860,
            -45228146052266052428497074865917240391,
            0x3fcae2714604428d,
        ),
        (
            -14674560818531912265366295646555938922,
            21851328767239664626048502206976698012,
            0xbfe57d734fedc376,
        ),
        (
            -77245014661929474536393179987819010912,
            15322620787503220680566598891207362338,
            0xc0142a3aeb24876f,
        ),
        (
            18926992102730643858964568247895455211,
            84489897809937144299024981358774765907,
            0x3fccac84cb5ffef5,
        ),
        (
            -23421882829770581849409045068450221101,
            -32938296725929711384829329194655868123,
            0x3fe6c132685c8180,
        ),
        (
            -79430642327411573913813166101006643090,
            -2700632442155838059134107931870718451,
            0x403d69701e62a8c0,
        ),
        (
            -14165485728598762438388857680160011211,
            -37425456942127459383987033606893925646,
            0x3fd839527dcb9789,
        ),
        (
            -83287175517782404074695570445980880615,
            11366623493581796152599192711827110931,
            0xc01d4f339ce714fc,
        ),
        (
            82991311424013667415349785581954728559,
            24320837629259026028023983107830022952,
            0x400b4c80564b49fe,
        ),
        (
            -43247580732925821604674172202572144497,
            69980555531283416426353481021816578021,
            0xbfe3c69bdf18652c,
        ),
        (
            -49824205143590304980549037594050525459,
            70627810952579593064914501122050587518,
            0xbfe6930669813c9c,
        ),
        (
            -49837716526090680749354541125735163950,
            -47497966813073077615748782176745649067,
            0x3ff0c9c4d9d0c216,
        ),
        (
            -43961861090386155765652880332591307871,
            64303893234428813147990089022617157449,
            0xbfe5e0866ae14256,
        ),
        (
            50969352479676330371188134524416818881,
            11187450253804332295204229801367560608,
            0x401239485812e3e4,
        ),
        (
            -5774054027669015680916733894414271554,
            63738530177678026318212125372955355898,
            0xbfb730e2e2ab1789,
        ),
        (
            -78969713091165838199885846172672828470,
            13638969696646424025096727436811882602,
            0xc01728f7450b2dfb,
        ),
        (
            4741233906603899899747195879419294044,
            -35891083905349855822196120301491601074,
            0xbfc0e8ac31a49f81,
        ),
        (
            67621672102692755948279800447770403849,
            -72104676482015920667296847663441706411,
            0xbfee02ac9883ae44,
        ),
        (
            -34317649883390907846709577445666236115,
            57129347843930115700920682146305159985,
            0xbfe338f110aa4bdd,
        ),
        (
            -72737769513966751917978737260358974505,
            -80283958938563204675885346925222208122,
            0x3fecfe00d801cd66,
        ),
        (
            42587812491866046361280249774098256522,
            -40845366113870360171901577364694102431,
            0xbff0aebbd1387aff,
        ),
        (
            49438930874146039544265876133853855909,
            82889380132451639191007185477019004831,
            0x3fe316132e3617d4,
        ),
        (
            81910416220562448381409274252172225883,
            -52096631363619624003191304277096350635,
            0xbff9280d8e8f6df9,
        ),
        (
            -34363236705845623639583724775488535342,
            1178340812014948279317329425668969177,
            0xc03d2992862d52f6,
        ),
        (
            22374815278219354256229720382591141781,
            68792329069971529640559935722416852917,
            0x3fd4d0ec190f5c69,
        ),
        (
            -15374503392709762872220613582089053239,
            51212935825733265682596141905016098649,
            0xbfd336992ff11c41,
        ),
        (
            51138451883148074667265042277928924190,
            -34212193167294733388646114774279603583,
            0xbff7ea781d74ffed,
        ),
        (
            23718576337804312244663701108259432326,
            70818032260457084713647479629042078532,
            0x3fd56f6032d8693e,
        ),
        (
            -42190736426308163197022494037763201399,
            14883748078790465978656520279570962919,
            0xc006ad6f4b4f6c2f,
        ),
        (
            -50629885502497124731063675710627479589,
            2070866236923480385071803223719126000,
            0xc03872dac78fdff5,
        ),
        (
            85053863447323324723657135279530046138,
            -6969810605139854156535372633865043086,
            0xc028680765f55510,
        ),
        (
            -4296662445812955013898014841460428081,
            3204172075065459575681998211667111396,
            0xbff574911e3e8ea4,
        ),
        (
            -77725655036020099617927458419557074971,
            36000584669329516283912417613708139913,
            0xc00145a782f33fe3,
        ),
        (
            -80902518737553588126325319582779662695,
            -28300871853393787187647080073652602149,
            0x4006de882da25fea,
        ),
        (
            65034808794640102214667277446949023466,
            25999675505883635185630753462164501250,
            0x400402ce48bc9f0d,
        ),
        (
            -84819889593668825911875460183395107452,
            -9148533746512685220098669765938966212,
            0x40228af76e5b7ee1,
        ),
        (
            -84408748357255735658607949548956225468,
            79239756633157018507485642562933525324,
            0xbff10b3106b22efa,
        ),
        (
            73421582468559091835762163816099568884,
            80397356985075903320290331045386937783,
            0x3fed39360dfcd782,
        ),
        (
            -52518828547700544855206389573313156475,
            61780089855865008988521922899500691857,
            0xbfeb33f675bd2d78,
        ),
        (
            -16208283655941921458594201903128961584,
            -59554716054988467587400598020925523248,
            0x3fd16b08c57796dd,
        ),
        (
            -11153207389043832069420638649931834036,
            45660385732207062741439928767839969965,
            0xbfcf440e6c094828,
        ),
        (
            -43742956169030574973780034397651477904,
            20679023255368754917731926184257184208,
            0xc000ec3214fb0fe4,
        ),
        (
            -30182030329203370014028409434842719347,
            -49908237850534749364225520537210472170,
            0x3fe35a1da8b53b04,
        ),
        (
            69279314446522563237708788093115798017,
            15722225825900340520339757265335340149,
            0x4011a0364dc7df05,
        ),
        (
            19105389102251772021590118882479731076,
            53762392414841893051201071988464741019,
            0x3fd6be55ee74eb1f,
        ),
        (
            -12688886560772729519913036674922015501,
            68289420686941997030649706362176794680,
            0xbfc7c8a2da1e476f,
        ),
        (
            39492326467383569558442623304192871790,
            51890343899343889229250292382035043475,
            0x3fe85ab53aacb195,
        ),
        (
            12257593366144966471950364196996696064,
            -42734702087120151304632452094356415277,
            0xbfd25b6c18629cad,
        ),
        (
            -59348512836395667078439080570946733007,
            32619952105581308981489323319874969090,
            0xbffd1c3bd8311770,
        ),
        (
            -56478019502364740976634801050318610359,
            84219754322484079317218512327273463540,
            0xbfe57594699d5a68,
        ),
        (
            9290360611709452283616166037660108425,
            949,
            0x46fe2aa867a3358e,
        ),
        (
            9062791023546346830085330222295653565,
            -945,
            0xc6fd8d60957eef93,
        ),
        (
            -38868705701533332350075299330457531337,
            -712,
            0x4725071385a1bc31,
        ),
        (
            73808539223026483205257221893280699218,
            817,
            0x47316626d605b8f0,
        ),
        (
            53408393310018530242910793224633042898,
            527,
            0x473384a7a2ad761e,
        ),
        (
            40970035576863255853480654668775951312,
            -797,
            0xc723ccf4923be094,
        ),
        (
            27837011627660486981392955976972033940,
            798,
            0x471adf8d542082d5,
        ),
        (
            -5895153945361484851652183196192590813,
            -168,
            0x471b085277bc4011,
        ),
        (
            -38294158241943130185237749266666676234,
            -503,
            0x472d5325f016413d,
        ),
        (
            -38861128811778301432313378361622428292,
            -33,
            0x476c5994eca49060,
        ),
        (
            -71212130173719058055982955702770025335,
            763,
            0xc731f99c61d18782,
        ),
        (
            50188798129078730991255201518500174294,
            206,
            0x47477610a6443175,
        ),
        (
            -47348391596625475805980005510640215004,
            187,
            0xc74861dc5db35c9b,
        ),
        (
            49884729938419748544812268125266099764,
            867,
            0x4726299a78bd3b97,
        ),
        (
            21440821126553552429497351044942893614,
            439,
            0x4722d002724c2a87,
        ),
        (
            -27171086058887070402149781892191117563,
            -724,
            0x471ce95094186796,
        ),
        (
            -18656990104492623466226459422508705529,
            642,
            0xc716633808ff23dd,
        ),
        (
            -42762651267620120321429066022748894472,
            643,
            0xc7299de3b8d3f93f,
        ),
        (
            -25972228826303476824585229290421037305,
            -193,
            0x4739eade80c5155f,
        ),
        (
            25154471110138600185196088222207769802,
            132,
            0x474259c5056f591a,
        ),
        (
            9007199254740995,
            85070591730234615865843651857942052864,
            0x3b60000000000002,
        ),
        (
            9007199254740997,
            85070591730234615865843651857942052864,
            0x3b60000000000002,
        ),
        (
            9007199254740995,
            10384593717069655257060992658440192,
            0x3c30000000000002,
        ),
        (
            9007199254740997,
            10384593717069655257060992658440192,
            0x3c30000000000002,
        ),
        (9007199254740995, 9444732965739290427392, 0x3eb0000000000002),
        (9007199254740997, 9444732965739290427392, 0x3eb0000000000002),
        (9007199254740995, 18014398509481984, 0x3fe0000000000002),
        (9007199254740997, 18014398509481984, 0x3fe0000000000002),
        (9007199254740995, 9007199254740992, 0x3ff0000000000002),
        (9007199254740997, 9007199254740992, 0x3ff0000000000002),
        (9007199254740995, 4503599627370496, 0x4000000000000002),
        (9007199254740997, 4503599627370496, 0x4000000000000002),
        (9007199254740995, 8796093022208, 0x4090000000000002),
        (9007199254740997, 8796093022208, 0x4090000000000002),
        (9007199254740995, 8192, 0x4270000000000002),
        (9007199254740997, 8192, 0x4270000000000002),
        (1152921504606847360, 1, 0x43b0000000000002),
        (1152921504606847616, 1, 0x43b0000000000002),
        (1267650600228229823709168271360, 1, 0x4630000000000002),
        (1267650600228230105184144982016, 1, 0x4630000000000002),
        (
            85070591730234644200042549075813335040,
            1,
            0x47d0000000000002,
        ),
        (
            85070591730234663089508480554394189824,
            1,
            0x47d0000000000002,
        ),
        (
            170141183460469231731687303715884105727,
            1,
            0x47e0000000000000,
        ),
        (
            -170141183460469231731687303715884105727,
            1,
            0xc7e0000000000000,
        ),
        (
            1,
            -170141183460469231731687303715884105727,
            0xb800000000000000,
        ),
        (i128::MIN, 1, 0xc7e0000000000000),
        (1, i128::MIN, 0xb800000000000000),
        (i128::MIN, i128::MIN, 0x3ff0000000000000),
        (i128::MIN, -1, 0x47e0000000000000),
        (
            170141183460469231731687303715884105727,
            170141183460469231731687303715884105727,
            0x3ff0000000000000,
        ),
        (
            170141183460469231731687303715884105726,
            170141183460469231731687303715884105727,
            0x3ff0000000000000,
        ),
        (
            1,
            170141183460469231731687303715884105727,
            0x3800000000000000,
        ),
        (
            3,
            170141183460469231731687303715884105727,
            0x3818000000000000,
        ),
        (
            170141183460469231731687303715884105727,
            3,
            0x47c5555555555555,
        ),
        (
            85070591730234615865843651857942052864,
            3,
            0x47b5555555555555,
        ),
        (
            5,
            85070591730234615865843651857942052864,
            0x3834000000000000,
        ),
        (
            7,
            127605887595351923798765477786913079296,
            0x3832aaaaaaaaaaab,
        ),
    ];

    /// The `26` fractions of the `k = 10` table for which
    /// `num as f64 / den as f64` is not the correctly rounded value.
    const RATIO_K10_MISROUNDED_CASES: &[(i128, i128, u64)] = &[
        // Wc[1][0][6]
        (-54220581756473969, 140359731240960000, 0xbfd8b918351c75a1),
        // Wc[1][0][7]
        (182950674298996883, 982518118686720000, 0x3fc7d598559763ab),
        // Wc[1][1][8]
        (-9072601277951951, 15412048920576000, 0xbfe2d6612abeb734),
        // Wc[1][2][3]
        (-1662254734370487601, 283838567620608000, 0xc0176ce400a69018),
        // Wc[1][2][5]
        (-923764987209375061, 135161222676480000, 0xc01b56920cd69843),
        // Wc[1][2][6]
        (378047083835329553, 81096733605888000, 0x4012a58fa5086a28),
        // Wc[1][2][8]
        (34675250583026971, 47306427936768000, 0x3fe774ac6881e827),
        // Wc[1][2][10]
        (219094447348157483, 17030314057236480000, 0x3f8a58f2ca922ffe),
        // Wc[1][3][2]
        (-1662254734370487601, 283838567620608000, 0xc0176ce400a69018),
        // Wc[1][4][7]
        (-20871418540789153, 4678657708032000, 0xc011d80c441fad27),
        // Wc[1][4][8]
        (9044275032176171, 6238210277376000, 0x3ff7327547f060f0),
        // Wc[1][5][2]
        (-923764987209375061, 135161222676480000, 0xc01b56920cd69843),
        // Wc[1][5][9]
        (80621498932819643, 304112751022080000, 0x3fd0f776a8eb87a5),
        // Wc[1][6][0]
        (-54220581756473969, 140359731240960000, 0xbfd8b918351c75a1),
        // Wc[1][6][2]
        (378047083835329553, 81096733605888000, 0x4012a58fa5086a28),
        // Wc[1][6][7]
        (-13266856114333477, 4678657708032000, 0xc006af5523bbce2c),
        // Wc[1][6][9]
        (-10144111113690517, 56143892496384000, 0xbfc7208a912048ab),
        // Wc[1][7][0]
        (182950674298996883, 982518118686720000, 0x3fc7d598559763ab),
        // Wc[1][7][4]
        (-20871418540789153, 4678657708032000, 0xc011d80c441fad27),
        // Wc[1][7][6]
        (-13266856114333477, 4678657708032000, 0xc006af5523bbce2c),
        // Wc[1][8][1]
        (-9072601277951951, 15412048920576000, 0xbfe2d6612abeb734),
        // Wc[1][8][2]
        (34675250583026971, 47306427936768000, 0x3fe774ac6881e827),
        // Wc[1][8][4]
        (9044275032176171, 6238210277376000, 0x3ff7327547f060f0),
        // Wc[1][9][5]
        (80621498932819643, 304112751022080000, 0x3fd0f776a8eb87a5),
        // Wc[1][9][6]
        (-10144111113690517, 56143892496384000, 0xbfc7208a912048ab),
        // Wc[1][10][2]
        (219094447348157483, 17030314057236480000, 0x3f8a58f2ca922ffe),
    ];

    #[test]
    fn tables_are_finite_up_to_order_ten() {
        // debug builds panic on an i128 overflow: the exact sum must stay
        // within its range up to the largest order
        for k in 1..=10 {
            let wc = calc_boundary_convolution(k);
            let n = (k + 1) as usize;
            assert_eq!(wc.len(), (k as usize - 1) * n * n);
            assert!(wc.iter().all(|w| w.is_finite()), "k = {k}");
        }
    }

    #[test]
    fn weights_are_the_convolutions_of_the_lagrange_basis() {
        // independent of the formula R(m, a, b): Wc[m][i][j] is the
        // convolution `int_0^m L_i(m - x) L_j(x) dx`. Expand the product
        // with the exact Lagrange coefficients and integrate the polynomial;
        // the expected value is converted with the same single rounding as
        // the table.
        let mul = |x: Rational, y: Rational| Rational::new(x.num * y.num, x.den * y.den);
        for k in 1..=10i64 {
            let n = (k + 1) as usize;
            let wi = poly::vandermonde_inverse(n);
            let wc = calc_boundary_convolution(k);
            for m in 1..=(k - 1) as usize {
                for i in 0..n {
                    // L_i(m - x) = sum_t u[t] x^t, with (m - x)^p expanded
                    let mut u = vec![Rational::from_int(0); n];
                    for (t, ut) in u.iter_mut().enumerate() {
                        for p in t..n {
                            let c = binomial(p as i64, t as i64) * (m as i128).pow((p - t) as u32);
                            let c = if t.is_multiple_of(2) { c } else { -c };
                            *ut = ut.add(mul(wi[p * n + i], Rational::from_int(c)));
                        }
                    }
                    for j in 0..n {
                        // L_j(x) = sum_q Wi[q][j] x^q
                        let mut c = vec![Rational::from_int(0); 2 * n - 1];
                        for (t, ut) in u.iter().enumerate() {
                            for q in 0..n {
                                c[t + q] = c[t + q].add(mul(*ut, wi[q * n + j]));
                            }
                        }
                        let mut acc = Rational::from_int(0);
                        for (s, cs) in c.iter().enumerate() {
                            let int = Rational::new((m as i128).pow(s as u32 + 1), s as i128 + 1);
                            acc = acc.add(mul(*cs, int));
                        }
                        assert_eq!(
                            wc[((m - 1) * n + i) * n + j].to_bits(),
                            ratio_to_f64(acc.num, acc.den).to_bits(),
                            "k = {k}, m = {m}, (i, j) = ({i}, {j})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ratio_to_f64_matches_the_division_below_2_53() {
        // below 2^53 both operands are exact in f64 and the division rounds
        // once: a reference for every sign combination
        for a in 1..=400i128 {
            for b in 1..=400i128 {
                for (n, d) in [(a, b), (-a, b), (a, -b), (-a, -b)] {
                    assert_eq!(
                        ratio_to_f64(n, d).to_bits(),
                        (n as f64 / d as f64).to_bits(),
                        "{n}/{d}"
                    );
                }
            }
        }
        // seeded random operands below 2^53
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) | 1) as i128
        };
        for _ in 0..50_000 {
            let (n, d) = (next(), next());
            for (n, d) in [(n, d), (-n, d), (n, -d), (-n, -d)] {
                assert_eq!(
                    ratio_to_f64(n, d).to_bits(),
                    (n as f64 / d as f64).to_bits(),
                    "{n}/{d}"
                );
            }
        }
        assert_eq!(ratio_to_f64(0, -7).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn ratio_to_f64_rounds_once_beyond_2_53() {
        let mut missed_by_division = 0;
        for &(n, d, want) in RATIO_RANDOM_CASES.iter().chain(RATIO_K10_MISROUNDED_CASES) {
            let got = ratio_to_f64(n, d);
            assert_eq!(got.to_bits(), want, "{n}/{d}: got {got:?}");
            if (n as f64 / d as f64).to_bits() != want {
                missed_by_division += 1;
            }
        }
        // the cases separate the two conversions
        assert_eq!(missed_by_division, 47);
    }

    #[test]
    fn ratio_to_f64_rejects_a_zero_denominator() {
        assert!(std::panic::catch_unwind(|| ratio_to_f64(1, 0)).is_err());
    }

    #[test]
    fn order_ten_entries_match_the_high_precision_reference() {
        // Wc[m][i][j] of the k = 10 table: the correctly rounded exact values
        // from a 60-digit quadrature of the Lagrange products
        let w = calc_boundary_convolution(10);
        let entries: [(usize, usize, usize, f64); 7] = [
            (1, 0, 6, -0.3862972754157799),
            (9, 10, 10, 0.00012549249690663132),
            (5, 5, 5, 5.692795200776543),
            (3, 7, 2, -0.03711620185133365),
            (9, 0, 10, -0.0022760121807017144),
            (2, 10, 0, 0.00024630869387222106),
            (7, 4, 9, -0.0157376478583863),
        ];
        for (m, i, j, want) in entries {
            assert_eq!(
                w[((m - 1) * 11 + i) * 11 + j].to_bits(),
                want.to_bits(),
                "Wc[{m}][{i}][{j}]"
            );
        }
    }
}
