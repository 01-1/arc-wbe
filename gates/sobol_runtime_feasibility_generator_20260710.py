#!/usr/bin/env python3
"""Dependency-free fixed-d=256,m=12 LMS-scrambled Sobol normal rows.

The compressed direction-number payload is extracted from SciPy's Joe--Kuo
table (bits=30), but generation below requires only NumPy and the standard
library.  It mirrors SciPy's LMS+digital-shift scramble.
"""

from __future__ import annotations

import base64
import zlib

import numpy as np

WIDTH = 256
BITS = 30
M = 12
ROWS = 1 << M
UNIFORM_LO = 2.0 ** -53
UNIFORM_HI = 1.0 - UNIFORM_LO

_DIRECTION_B85 = (
    "c-pkRdt6iXA3y%ug)z1<1`HT5-~a>0ecZ=w91{!8Y?>CC)lw@mvT0_RX2(lPhBhx1nbk5sN~VU_NX^gjlHz@NMKhh3l+={Wip=Ww",
    "Jni`X^yRz${P}q7!PptS-|yG`@_fA`B8G@`cq#Fc<0ZvQLd4+*OaDJTCh?I4FEd^hcqQRgfWOQj;;Zl&&n<Xv#`6k1Pr~zp0^*iu",
    "5dT*d{+hvGTkzLr{B;HXIthPWz-4e5d=+2C;GYHm%=lM<e@XaPfPep3*MOG^uQI%%ux>JToQKCTc$`MmTuwAHf+z;xPqPqLZXh<o",
    "gl&~!TTx_aPA1dHJSvNcp{TSp?&D46?3f`Dc*@|Z1y2olYQod9v^3$r|9{pMNTJ8ehF2Wc#rb&bgDEDm2N5+y60Iu*rQ(TX*~D=S",
    "v0E+VO;(a;pq?6U*{J{cI2xX4rneq2ahHB8X1^I91ad@L$X=?XhIl=#%eIk}i6fb%2Me;qbxW`={`)M>>Bf3pYNA@~BMWq1jD7qV",
    "OI$5Oe2$L1A1TRsBa7-z*l1mS3C*H-Iw(!!ewY@?kMFH!ydRl$&KniVx)a4&>*`}|vxq4<DAoOAU4hOTyz=qF`8d#&#~v6f9vkr(",
    "{MG~zt-)S)mk_r;oA_%<`0FtIbu<~cVAAmUl+UOrNEg9HG*|JqedUaGUV_eeBGa&ZWuWG(b7^Y=*A(qe+nv)xt_tzO`6vbJhG1Q+",
    ">CVM@L5s#fq8&Kz2cbk~qlpWRCq6onm=E&EayFe>LW}8UbOaTDkVX+_HQchfrTo4V+3skYvMVlBuEZNT-q`Ra6mR}<UbD#GB=Niu",
    "dlvL{<2x>#Q3(~0M-$BgFZv`9*HT67yc}X)SCHrKFsjSWqYJ03Bz0Gk#8tvcl^Lu=9?VE*X<gJOvW7F4uVW9ToJ>rQop9(_k)L%{",
    "c;Ryna?FChj24eUgGR`CHTaN`N)#a_ZkC1EzNO?|Q9-&v@ud93MmvV(k~Bj>l87*F>MR4>t8cd3y`tD2J}9##DBZN<D`UYqsY>|-",
    "1<3!gu2qz)RFTd&H;?taSj!1nYQ!FPfcB%olTGQwoz)Zjii+HGq|~}cL2VZb>C~$+B%M=6lD!$+wC}_CyCzgQ7L&ogMp9ue3y!aU",
    "UlDX*@f`WnYxWfMIPV>k5{q>~TONG%LcW}tM4bsl<Dgf=!-&?yZb*}em#N7yErhzh(vab56lG1e(C`sf`f+_07c)JO&DvvNhM%o$",
    "?Q-UJH(!g}clDL3$ue(IpwwC3Lyt6~Twxs!XXZh31~j*0FD;<Q0`Os0IMJ>=;+7eRe^*Z4>9OS8S4rljS~AVerE^*vjZv3yEmO+*",
    "3+>qs*>Xjz?49I}sUhZiKG>LlP8<K$San(tdA?&eV5>N+!($&_8P>&~>adSRurpH7JSdR3*K&y8Urz2P0;ub~H0n^qlPNfy7EjaB",
    "PumRK&?}Yfcf&Hh5^tPEVvkiyI*Xbl!$MBW&Z*Z+AI|b)Cm}}!`^bR?Joe&%?zo`4R@mz5M68Sb$n?ZXB8inokQeqvw>_GaXS1kj",
    "kd$^lBBM`^skxs@16h5aBK*g5@E?~Lh*O$y4lOzI^T-w3gU*6qz#0Uea;c)dg5JWyQVsZ94LT%2$Ho|m>ntPwr8shT#*lqnFm-l@",
    "(YkSYG_=!7=f|74&CkZOo)Q`3-V?2}2k4b`oT6e}?~;sL?vRMJ`I$ZJvtT!h#q+{GusEM1ov0N$T!-(jfNqU~PJfm`+$b5bE8@xf",
    "OcB{X%cbs46IK79rO~s3>BC{+T!%N6FZwZv89gi8^5L-L#tv`J&7vPGBS+7Qm3%lX%+GlRI>R1vm7p{1gcoPjf$uHQtJ~o9a>)E2",
    "ST~~zW>iIve!1kjCZ)D1;iTEFp&3s^(tU^XxU73a_%&C<m>E-Ib@y$zG-i!3++O1;%$cz_KjOX|b&vB3>*7qXw*-4jEcoESo^;@8",
    "8)$JGv@gTj{{??fMiPHjO^nP;c8P|%C1G?ymO_)cDEd!tDfia7IDV)-nVEdjWckn4_{IpX<n}@~zwEK|5$Qh<O7LT^1iOK=a-b^@",
    "`|XAAb3&Gt;Ijs6Uj`pui6G7kJLy!Cw^2^616t}(rBj6_n3g^hPQPys<Fr%M{LF1O=I*nhx@X7cwH#9?9@(S~Qh4j)vRzxs|8d?s",
    "_Db*-uuB|l3XA<XKnExGQwtp)27Z44nk-93AOrm+BYC0kt<bwRE{-bL1Uld?r(PC0muOY-1G9=8K|5{E*M65ajQzm0Xuz^6{71AP",
    "SXGa83q?AI`0ON$^&F6IC;V|I?8!LrelpI!9_Qtx#Iq*yI>N}~lu=!$hAPHMXx5*BH1KB$m+m$5ah7oRrbfBvdwFp8A8)508MRd}",
    "rO5#@TA$Lx{}zgJCD@Gw(J#OsTM&wMDu_;j7E2+E*AeG%>BNHn4tF`$RgtwOg_>6AX!5mE8q-_G)m#hVt9#4czmHZpGi4#o^A4nJ",
    "KJ!AJlxAopbVl04UYSJu94-2zB%dC6u(lHGjs(xo;q@i-*AYa#M@nvG4!K5#kn&tQjr=lzE_=hNuZ!UV>H^tE)>SgP%ek)ICvqD5",
    "NUb-^vn!--rB>z|S>;E6!^HE#F0(kh13z7`mo4D^be#Ve(0(lgxyr}7p~Tdf$TK*DYHSJAv@Dz+v{lh>%aXYlZ6$2dvMBdZTZ-=5",
    "vcit%w&?byT`H3@r&DB}mx}$wb;Y7QgU4)&Pk-%r-wt`&480qR^G<{RZHGVF97fCy*sJYYs)gSuToFL=Lsj(E^SRvd8yYrfyNr3x",
    "T-B9%wXo*o#)SU!tKy{b8>?hrRb}_kqdWdLU5x8+R=4mNyl=psjG(&;`}i@GxNa%2!!pUeJeyk2!Vf|ZF2KH?gk67<wQvg=E4#xk",
    "_h#0roR9VoX_<4VWO~{knbMQ9iX>02HuT`H;0M7+3Kr!V_Sp+wyYL+=_Ev%K4}uH~R3V;J!*1k~dsYBh{;*MVekt9YVxi@r+jCUG",
    "ouM3d21&e!I8)bdHlux#qg-C>36!cjf@F`bu=a>I3PrgJ7c&je!;L+(VqZ$w&LfcPKe4Z1*ei1wv8GJ&9xEhIdM-8o8BdyE_<iIR",
    "DJBhf`oUuUv1dY=4^D^MxBQXaa6iY~{G6?nPIVVaer}cb&|kf%zrwok1$?D%T@UEp4V$qKcIGnZGO&oa%Sz%029V>jg<JzeNqISg",
    "P7Q>5jw+$dxm8@vo-+1!FO}n9eylTmc3yjZM?vAvQ}W>l^UGfdpDpR3za}w$7V-t`fyY}f;yDlOv;p+q44J!#b3YC`KL!2$&_>=F",
    "rR2HaNalD2Eh#9Wmz%R`*`!o1RFTFuzFFjbYqGR;YH~zFhS$9Beoy4Nx0)kXPn{IzCw|TmGdirrL!RA`V<8^u0Bub0!^g1p1PyWH",
    "VXuM;h*^bQXNOXAw~X`)N@@J6RO+3r=aw|Y^IHa|dvoXH+JAXp-6rKK*H!b(xezAtfs3t~J?N|v>5O$ar7yk`=xm34HRAI{IP1&c",
    "0sI|z65+#qh8%~q)O9eKlu8vDhZWHBlNI#qd?R=JP!_xJpxzzdXt3B9No(G{Ra!ApmbZBFt?~l5tf&5J{t-VXiFqja+68;jj`Oa-",
    "_YNTbni))-LPvb2mE1LQ>Z(_ea&#44_&kb|M#s^W&*QmO*DCo*FB#kk0}7pQ-J>;JPAyoG`%T<pM1U_${@dU7^n(JOu@(op<v|aJ",
    "kf*>eSOs~;_vT=)GsB514kvbgFfp|%(wR!gupyR?Jb-oE)by;riVM7!z%ws}I_A!?xoT%db=)kD=)Y#Zb@<wjq`3tbDtqXWQPiVK",
    "G2RgDGvv&Uc&i<9bsPJP2i?DhKiZ%`+@3`a_$!M&nmXH)Y5MIr`tyT)dNowV{rABj_Jbc{-3Qtwt#9A9)NYC|S;u@`xY%A?y86VI",
    "1wGa^iToATg?#fkrvrA&j-PFi@e0_<YxtXq(BI;GVxtnt;Y}rvOGk|z<utP1K&l4|Df6Br?!DS*_O2CD$Mp@W)+dfDY6o&;+|A2|",
    "zvie|z0jW1BaYRJ_NqjTV{tYHe7EDA2Jqbk`9238gdn~e8bSP~QuuuvxuEY2oR*pppA2=!(-oJ38{QVi?p&w$Do-YRo}5_Fnd`Q#",
    "xa}%Fx28>XIDQ@ZiLY#;oy0y^oST8)w?MucVV7@XU#AgkKAHk^mP2f55i#y8vN)?qX^y6frVO%;S5WX=8P|Nw#J<u|%FKE!+H${1",
    "*Z2lmZ`YO<o=#(=kD4q6e(EYVQI7<_kF^=-Y%5}{I?!Md&U^s8_(@LOPld#uHWJf`y2`0|GG9xfir!_kB{qpBN{hIv85-7nrkwfj",
    "ki6DDt0U?@84|jv*Xoo3YloDswXcruao#L3KT?ZvyO8h02e#t#8pw7N{P9zewPg6CjcQ`OvBb2OlVzWplsjdVbvT#aK2%Egp33JG",
    "Pw820at1TAETxro2i3%NCN&jJ%DSl0YDdTt68x-NEaqiGJqNPwhRtw-7A=tLQ_%0nz~d<R-|Qe_XU7n8T1S?LVyQ7Wfa(#4L>?=k",
    "*x&WsrxWwo(ZxFV>Hw?rg;0~Zl!;jsW{LZ=qEY#D_SM85@_Z*xQTysSf9Lx`JqPDDfc7i!Xb{f-4&n{Oz-+UDn6b&Eo0(5-qs%nP",
    "6hkUQGR3aQ;&g-J*phdm+=~vy*<b81G#3o2yc+vX;EgGVQg#jRF!`~Qf?og~P$wcj+*jWa>~k&XaSD5ifnLSN5w|pu__3&)-K8bV",
    "E;Y4n4yS$3<k7s5v2^`KE7x{9jK3Zq<_=(TorYFJM^bl!F=ItsS@K<`PcwFzd(c@f#v8(U;Tu@Uog4h@f^Ky}rU!%mTd?l0S;Xx{",
    "KDZ~4m<Q9y@^J+<{w$&DSBx~HvWO<WTg-j3*TCMh$Lh{HrtB&`Pj#-Na{WX5lRnt=bNW5QUorQvlO}OpAs^I=`5?~j!K+=clQ{2W",
    "_^D5zM>kP7Yk)i#TS!-FrIrj0jg%lvRTWU5xv|`gbLs2@N3HJF<H$4F8dH-JnY$osW!_`cx7kYT`vmyO7o?(n#(OqLtRsQmwb)Aq",
    "_^iU2*Q1WKEsNM6P`|o?I?@C=wIN5YADK>z%L-{=Vmw#K#IpIViSCGQ>WbgTbd(>`P0HJC&S>eB2aFVA@&C(S3H*&0>qtU<Bh(jP",
    "d0;=<p}#w@m!YuX;mA`KXo!6g^_&G6<a|*>ZF)6LT5q7G<1*>R|5S2QP9^ap$5c9+SLt*Qj}K|dZ7DfYbfok~_>JIQsk?IhoL5*E",
    "YjTh!9{WO^OddS%gq^Ae&tt&vIMg@ZRuWqmf-+(-Ia}hXd3*&8Ym?KA$%!;ED~$V66U4u=-sA{zgzAEI8IAGEl))7X%I3+(7WYkj",
    "H?&8bBNy$4AkSEf#a<jpqq^XaTVSjD<J_0Q|4QU3d5EvnMsi0ZKMD+@w)weqWRH?^ROO_eUCeEnP{p4AwZ!|{rr6ePJ0oi(u#+az",
    "Cnb`0N;1XzspsT~b_4Weaegm;ZAJW3i+x<cn)ATxZ(%p)ClbF6=as}j>_W*rFOH1gmeSJjBDyp+mz#ICkUjpH)=}6muS-6mpzZA^",
    "QV0KhJ-T0BM#_YO*VB5?*(lZtK?g3($6p8d3O`5f_|6XaLc|mF9_0BUBhrZ~a(|Id&dJ4O(5fg)9Y6~|)YE-uZJg_PB=7uA>izPB",
    "-2U9P%$CZ#Rf8=dhSdLrW_>s^Bj#W7lq#R!7xD!!WIzXfZUeuELDugA&mUI84`vd-qmbNgC3S60A+t4=OvXU6El#A&zERv8=Tg}{",
    "%|VVe!wc*OIt;a&rzZ9Pq1}4x<BeefUmvmb;IIB4b+Z^>UBwICu|RK?c;AH2w}B3ER^k%n#3yUXt*#=^R1H}jDv~=*q^(m^)$IgM",
    "vs}We&K5Y9yb<o)yDbt{r+9NHmG8;qq;mt=B0qJrGLg>ke=OEP{!TjV!GLuafDZle`a=Tuj67w2F}b(usq1q&DdC$mx(wLoAPQKJ",
    "$!Yrqu<}22-q)Iqo;`=-x>qIUPXA?D{N2=D_ML2cL=XMV65|aael8aM5$LbOUJRgj6J+}s^z1z1NlO5+hf#liF_rAsa!JW3NMkpV",
    "qOFwFi}ajifP|&oVE62uDK6HbH!uD}K0W@UlAX!#g`OYwbwCe!zEgh|^cOs3vA!2H)}f}-2-&ZOJ-H0JOCVS8RS>@wdH3=va^6JV",
    "-B?cY>S&Up?h+u$;~L+K=XY;S@h;9Nv^&>n+oW}&I|2sBpOX*I9vIli>PN2f#rj5wuYTpgIy(4==4hY2LLNkYqll}@BfcPqysM(g",
    "GRjCwgPb&S6-ia$6!303*R;#TZ@*;oo*$OtEKajE&s>?fgC9*#jg%$j{V6Z;)7OX;`76XZ;HwwE>+ros{M-Q^Zvg#&Pes_5OZ>Ea",
    "V$Pr*yCIL1b!L*oZ%A!6QdgC6uchkP9o7Wzvd=1A8=lN*%RW=EW5R~u#UXXYV*-yv`N_M(#JmhN;1O3jkc+xtOWI*GZsI#TLH{S<",
    "_f^HjDhkLu1>K<mR#K)GlRPhtW-X4OdxuAJ7v40pCEsK-(?^v%`@J1pd$w=%63fPl`~Yw6D?#>{9_v<!elT0i2PHoJb>Xb-T3_A!",
    "0BqO9D#RNp#8yO-_oZxd-XkUDszj1!rqirj@pQkTko!JV#ZELNGqV>4IIC+4y3dx%mb`pBJAdk;=tu9qC%})rvWfX`y04z&hCI0-",
    ";~n_!ZO~#1_7{b|#_R;*?~XvfOh@*;L8QEBBF(1>G-tk^c5T&hpS@VfFBlW&Na`+f4KB&-q}swIs?Ek<D8A^`z?qT%VxI-MDiQMq",
    "m9L&t16^7IT^o<Ry$QM6n@0QuDR~#dd_Sru<(4RtU&*AYk217<X*T!lZ(9BZPnq|-Tbiyf_N$wf<10+^Hv_|F?#=yv=LBsJIv0xP",
    "4e{l_EY8V*-Y&>^JM{J@?8s%rlb6DXdlvaZc{DMq3UcXV(B_FI`80-R?M|lsb#dHhnU(wlh79j2b)suXV})|r_Vgt~CS~O>|0KQA",
    "{!NLWdYzz0;3)@M^AWy&s0;R^26n0*vF&C2{0RM&-msIuhLN`@j$FnZtPB3WpF}fy<<kC9<y`$uHUG$lSa1D<na+U|CCZC`DVMyP",
    "tsOM`_2k$+JLCN5QHGfR+I(@l8?t1_xm)m^!T8P=ti2WMjznCyC79fMk%!)_qQ;SWs@@`}#rH;1>^cKCwc5n?T@~bBx!vf}cbQw>",
    "xKVJV=GmNC>8nd$d+-~7{Ud?rfj<8$#IX+0-y_u9@xDLkum!YU20OVj32q~U9H=iir<GCT{tBx8k)hGLLYf$o!Oad&=ljekVZI*{",
    "-Bq=?qWMx$`nvqFaksvERG)0IMgL2mE?DftlCSP9#8;g{I0C(z3O)M@dG}S+`QIud?|~w6cFL(yuBPg_Rdl(3G`*Ou<)}`|2JTQY",
    "G+b{9I9k%&bRoL1{}gr1<?75a&eBXj^{X85yo|3O>VQA;B>LiK4eawg$i%&{S6}22`*|UGP0`f#Oe8f9g!$f-LD!0l>5p?|oUGBr",
    "2b{7pikB^|(tBcSU&}PF>w7!m<kNA=pT3MY_V9yp(OzYWaXVz!fuDBBS~p~04*80Kt$rVU=Z6Z2f5uA8aRWK68Pu4YO8a1LGEmzY",
    "v%ZScID&YEUExr57uuEb><;-tX0YPh<kQnWFZg=*gj_%Jd?)`^`06TN*f0-d3;vuIh4}KLf+XTrpg%Y=nH;kV$kIEG8j+J%n-l4&",
    "k|?UoD&_(=ma&1)7rST1OS<lwX>9ZE){lF6qU_}>?<Gey)CBbKzjCpjQ{{`>|L#kM`0|AY@O=^F`6NF7-b~zmCB%M;x>=o-THi~c",
    "#vw-PkG}uK<-xT2&wNfbl;LIRT4wf2rnP!vO2b*5uCOi8yycA$<2(0iwSM~Dcj}WuT_wTS{}uG84tu%{+pz`fAB7%$iaOF)`6!E*",
    "l1rXUjc*mv4v(5HE)J&$21aq6H#2y9TZQ9^txT88n&0{Q8)1XD4Kd%Udm`Y?cj|iPM@CVu62y2CG-d?-#dkX4yDoqS<MFc``NDqa",
    "Z&n_8_mz<6Kfo2d7eLkbB+;nv<LRB}0=W^1YJUC+mHVlx1k0kG=0>kcapY95(vywPDf(7#w)ydcNuu3=J?0@O4xHN#AA$ZhEx~?P",
    "f$m?X5$8s}zagC*%Z%jAPo>6d(4+T?>GHt@Dt9uR_h1!U>a@789*lApI*lFQ9W2sYo#}h7+#hnn)++ZC=Scr2S1ja=foywVFFWy_",
    "LVPv|G1f}tp@vdo;Vaz_=aGG5DK$<_rt00rlr%AdBF?9AEvqH$LvwQ7Z+OhQ{v)+DEhBOY9~fba39c*tRy{7M2Y)NXb>Z*1QlI~2",
    "us1u-SqJ)7;Mv{S-*>QAlhLOuFD6Id9I|(-sIjY(c8o2di=|=oVLu&L_kA8aw64;@SwozAPe(OgDp4<ZdwXfw`BL+L7xoMAQ{T{v",
    "`YYJ00$;ydz;AS7U-h8JM*PjAs6SgPiJxvG=4%ak#zc`hFON=5f&QMzrQ<(@bLr<b{I^b<`{PX-OXv=3$Bf~@`?jZB&b3GqkH`F<",
    "yzHO(e!SRcPY~;(3K5@!-&a7dMkBu3ihQsF@y4cPa(9<e>-!PZ(jleIE48#7y`(R<CUYxoCVrEp)RD1VV!3&#pz{r*x}xTdw4^1T",
    "f_GNu8U6GZ?(|c@TOPiNfjzcBrp>UIoAKRSp#K!qy$>Q!S*aoS?J8=Wn?NnY!fEppI$C_hME}hW;XYd`Wm~<G?xfh#u9rERS$#6-",
    "wrq9sx!F%^4)6L})kB`|=&wx72L(R}eQHhg)nn^%<~?{e5qh+(lz0dF`$ICRwWORnpG&26k67v4vJm>5(R0JUGxIM|w)-_#u)Td(",
    "kn)))D=NmZA$QNR6kXc!QktLqNF~Y@d;^O$(Ep+?eAEEFUI#lJ4|=F`iTlt>tS1?1jEQu+a;Wp;QqoV;(6z5hDYPPhtFNwPAHQUA",
    "1Uph)A+5G%iNU-=EjKDO3WY=(lIF+nYsGpF=*+^$GdMSL5^BS<Ndlgs(wF}NJHww&BF9<y1;ovDzZKFr)TKsUsiIf=>bUPeG4fxI",
    "*SO{LqICgZGc9kN3m(SbA2;J%a!BtjpGNqp#}<qAN$3<S^t(Z0C-ez@8`^>M+8|r|p}&R5yQ^U*UnwG8ik|AWhmx_yM%Uhv(TP*#",
    "+_c3Z{H|b|`$ANp>)S<&j;VLiz8$aV4t@9EgfGt+BK`CwBgOtc*5rl$pw?GsFw1@Y(8=KWAE=w9qmHy4e(<>va*dIb`KKIOgnrB7",
    "d1Z9;0pez&-O_iv(wqKVTG!iSm<IDt)}{|O2i{m3mORljU)#g)%SHQ~?TfGe&i6s@F7Uby&*kvPZ{ys%WyFnu{~Zl9*>7d!O1F{u",
    "xP=y6P}0(*bQ;pThzoAhvRb*3QAL(nv_a+8*QZ78JMfs~?W~WK`xnIp`;n_05m!(s_V<H)eGMykqQ^e9_&FDJemsQu=hMjXT{PK0",
    "O{BU9B53evBV8MAp}CG2F2Iq_QkTX})?k-ZAJZ+@mam|>>BY0ols~gAx57_d^p0GCuPkhc8=qUi^KRHMIryOhKY%CVjHvU6p<lKz",
    "gIc#&q2CQXnyRIXr{n1N?_#+CGVno^<qe=>yNXKM1GwB%N;YDHhKY_-IL&_QqG6(62=Udg1iV=*UY$7q0`PqseA8OU^KX>^Dxw~%",
    "sGwG92AMSy+Wb}=Ej^>Azt#YMHQdGmSK)X)FQ}`RDXuO(tKih;6#b!=&trc3>!QR@oj+B~yP>zN5N}9){rwiany`;;_+C2v!T>X|",
    "KSGbbD@Ocmq`K=7WPBoyE+YncAVAB>RGGZ$txCt6)xq}OBMS{bBr2K)dQ!$tS(SFZ&lKCgKr4m%BxD=7fxr7l7U*%U;1_V-V$kFp",
    ";QGD{BYuID9QQ_%t}BT;e@UYwBSWFyM*8YVI5+&$NH&@y_t^o(_M;7o?#8vUQ#-<K1FjpYj(<DfA8sUG^ha8sy%KOi7RXl(?8PL|",
    "VL9TSXwX>^Kzw61d5`Boe<4=`HFW9+GtGJ`nzpwoxbHrc^AQi1G4;QP=#Jf#v?;t{#sH^zPk=e-u&O`xkSmQy=U5Rhif7&;pFa|C",
    "jD>jKPpA{ZJ|9XY{ulVcjY*^f4yy&R-;rm7X$JbWyUJDEGJ7_E^wT`%4TGv{k2<^Klg6Nm<Buz@&B!m<x_@W9pL(oH#1()RJU(~h",
    ">`r{I24@`yyL}RT*i;BSQXsJl(I4y|PL}Z@WZqXmlgv7rv!<BtKMO9&v-ydO6pqFkMOV$`BIT;-X)CPTR2$Pb=p?V4kM6OqQIzLc",
    "F&_l28Q2OZo;QNeM_{9qaNb{$@AoetwkwR>r-1ufjy}$sGWcIPX;EiTmq@s&^;*6cXLGm*Tb(~oPH#!kg{)Y+uxNSH0a<$g&y)T1",
    "o$u6Dg!l@)bz^-ec-stGR71CbL84FciTfS)>WqdQS8}LpkcB!|rBmTZ1!?2cDfr!dZd&hrcK@fT%*Y?I?Y|#Pta<K*nEp2-)d#+J",
    "WzWvrFX^Gb6=MEd>g&_7IHNnp=NB5WUNz#eBz&g?_4bvh*LhROdo-N7ehZ<xlc`iOI)W6a8OrX-=BEB9mfd<G+ueCQ%JuV~lE%wt",
    "vYURrRiQffXu<y9zcl!<&tYPJ5NGFMHyO~}3B78DEg6jcOvb)qGl^RW{KnBB$d!h=Y%*$@TttPa(JSs&QNUkvZq_L!zb_-sz4EDw",
    "E~`GKGtdxQAGAQVN`1O0DDcym9{er*hdirAd=6raF2vG}cy<KxFdjc2u0T62m)M(0<b4<BY$xpGGr4pd^#kqX0+Q^n<lc0a@%KKe",
    "c0BuevFFF7GV|=zkQIGyhF!k;u3`83ooYYvRn9;1!75)})B^jWge)|H&x3I0AA!$V6H4r(CFJ-i5b1=KI%DOK$}o~Dvnix0m6IIO",
    "@j<I|y>ASX>-s#G)I54($}s!Yva)NRD6Si~TKweQk)l7sPd3g+=PuCMAoL}%?rG5ZEb1!n782hJ_`22TOP1QGZgd*m>?0*<TRus$",
    "0=STi3@e|Vz;I{sbUP=d*Nu2EGwZK(tyVTUAyzif=0}gpM4YP7|ApQHUqu$kF<^+K$6loH#q*Ju-H&xQrX&9iCi|a7)H0T#MFsJs",
    "<y55R)tu5B!)m*N-KsiGm)2C#Mzabg$?r*AI6FJB@6PBDKXy_jo;S>gLq=VhTCpD`UK;F43i*072k|8Q;MsU`7otzM(?s2u6Dezb",
    "C`q||lJQCo@gOg2W!$8Uw3DXFESZ|nB)dOy%ao+_y}brXd&u*hcrxvOal2RNvsb`bksSM(gYWdQ0qy|&MszWGt8~<rEGKJx0xenx",
    "T&^REB&{i&tW(2E#+luu$aPAmBzI5Q7kzvC)pYF>KWAOa8<5`Pym#zXpbyvgcmFq5#PwkxO`y+R*k2m>doz~!-NEF2N<ywjt<*lD",
    "fKCPFK(10q)@0&nTM#dKx`dfBxYWu27TR|1K+c8vXS3Gq4i9<g^r1*Ub%rppZYC4^M~JnYAwGGY4j*whWbFs!N9*7hlB3D1*OSf}",
    "M71g{4GWNvv@wh%JMuZn(o9zNZ80<DR9V-y;~J~?Wcu{*hx3k|Y&QJ5Xj76OTwkgfZ@`A|*bf7J(t*Z}u$}$!`!(#LG7@moIAR0Q",
    "mz2ejB`}TJfeWtgRYlUL3?v(aIQG>Xw)d?Z$Krg(<J1{yI^Wc5Ufg2{+mjW1wohDzpT0(_7(WZ=1+Tr}wH-dO5zqGF%*&zUMZg1x",
    "Vs2s->_%iJxxO<{!-*(5au9X?#YH6hB9EK;jfUNNe}-fD^{Cbd9*|m_XVa4E?E#a!TT1>Dv0m;cKeCB+l|UapN9ar15ofex?F)!4",
    "hvN6Ws9#+vA$~|IdHW@jCB2f`)>hHFNF7bR5=C2w6m#eL1@V^jJnyEplCIAqa~q~Ut6Q=4t%~^b%XLQ!CeXj&M$&xtiibU8un!m3",
    "Z--o&urJ{EsTb<)=YR(;L7ibPL(X;`)vO4m9Yf?ab$Bvu{WX#svATjCTN~}}dm+NPbG6AZby)W1trts%dpD(hzvi1_Kk}R*`az-2",
    "4|;f!d%Liw8a&H_Y>$NwoPpiA2L3KgBWA0NT7fCGq83s8eIQL;WTW1Fayj>Bsr-*+CGMCXLv%k!rz*dg9@sySE!mRzk+kn>OHdEF",
    "l8gPL1RtJ($2xAkFJGvE?AK%M(>VJC)V%|Mg9?EDLZ+NE(vTm;(WzBxnmW-$+na*95qp$uYi_Wku}spn@(HW8?r6p4(-SRwAU9Pq",
    "KGRP<=MF9i=i;F=4)C)Ja$W-(nBWtJf=0k;a7GoeSl<CmRM)m-YOd4Lq8BtYRhve==70^o^7%C_QA|U4xhJ<ftGS_Xz^Savs>t7l",
    "q`X-#_?G|24~B{B2Kn$Ri1^8Zy|hD@vcU65*vcU|uLbedeW(*Y8BVU>imC0o7IsotH-~!53%IjysMy2%Rm{U#<(|LvnawAUSf_7!",
    "(Q<lITG)uJcgYV9Dqig42zC-R90q%_$N6yPBOzxt@Zr@&v;ok^NkQLgXc#%Smt$K}8uyrqrtX(guT=@$h<7Y}-gAZCPrguj`V?ro",
    "Gsi_vlE1Ir)^9{<M3E)Chg_NdF&C5U!>I~!j)2>20e_8<?d7nUZ=v7)R2i}8;W!?PB~Mc}wf9d)*q1}YPiN5KeKA~ny@5S8$;do$",
    "Oz%3(7q#uVrd)Dty>XUzV`%8Se2X7kpG~xr`M!0%av$!i6LcPi=ffdua}jUsN53pXiaZp3$!{ackQz*K$W+q17J2~jModl!8~9S5",
    "H!-}tYaqq6yekP>va3n5YRazS|M1IV{nSNmVq9nQ^<mwDeTF@40WEGqwy)uD9#RuG*GB9ZGco8(x?U`zn$&V?Dh#G1X)?{7Qpxpg",
    "SMW_E3cMeM$GHx*2G)(bD`Q-IYDn3Y%G}R~QH38piV@dM^x;MvLLLe^uM_G>c)k>J_CESnsHL%cQ5RhiL9XT~>i(^Sj0Z5Uv80&h",
    ">qEHTOY&Ibb&Gq%jU<=nE<^LXSG0u(Zs(;o?6N-am|%bYue|Jz-3S)*Zt&Dy<ine_z-}J_&qsj<Yte^=&1PFd$vqHw2GmK~F@sV4",
    "Ne(SOQAJ-rFX7rZnfX4a<K4`aK+oK+wDv3A=}V5>Qr+4+AvoghFZ6zJGI!?bB7OQR#O)sJ$&B^&p!rnnYd`X%#ah_sOk$qQrPjX`",
    "RI|2>s_i;Tx`euSekivm)W*&T$YMwuX^{q(c21irS+u=RdQ9?jm0zy?J=jlwFh;CzWc&Kv=>L&`!)t;6GeW);pphDWUkUsb`u*Pf",
    "if~>X)qN--Ic(OsXEW(=S2lO$ze>JOxXvx(iaZiFr&hw`Uy#@n76X7#M8nGb@Hgil`~_XTLcI<@>!8<@U@v##yelh-8<I@yap><m",
    "n46dsNgekD(q>@m&wY_Z^M5Gk9w}6@x057ZZhn=EeL24QiMz_j9lDZwpuDx{)9jHw^P@X;Z$Xc+J}>m0@zVo+HshRykmWz&Bcp(0",
    "yvq!DWjx$f1$B*2q1rPji^D!I#vbwp%ee_psQ3pB2By1rOxHUZ>2)X8+v=ZciyJVlLb7t}&sl!zW|1NeD#AAh#Nej|=kJ6Z8)2(X",
    "gZE>AXPAk6VGrp1aV*wNq|U|!8n(57hA#=BwUbJ?AlS*3_oO=B7%8(~8<^BF>WhM=_@&8fWRnA?(Ued>em`E!kMezSj#rrDgFcyY",
    "ZqzYo4QzA*;*CTb@$eVSM&N=%qp2e&h!%`d(Bg@4wED+H?p8$(e_^oMJ8`DOnR-9d@=Z?qz6E0<uPvTfw6*oe?0=afi1O7L{`ULO",
    "8%v6>udxW<3&UO?MIUF2g80Ry#N5_V>ptLB(lcn-=aIB{c_O`V!^o+2$=GYDRg9`K#Ch$Ksje}!;&ygNa9GAq!6P2K+w4c4Rbrh`",
    "ST|MdOPYLhsgq#C-p2E3z{j#E1g4-)IKG&iALdZ^6d9R%h0$WnD;1Q+bB~lTd_;MO`{mE$bt8Wbt&^P2SV6f}OQ-x<@IHGo#7~~0",
    "67l?y9~RHuc<vGG73jMebbk!5k8$1(%%zS9A;*SX(jC=M<LEq^{$C}XtJTm0pCob5oX+EqUbeVrbu+HloLLQy4RM8GM}sonxMqC+",
    "wZHQJ1&<UZ_HlGR{ncR)b>Q(N$W<iHejan08&im1XTjJTa0M?Q51p4xN8GSiUnkPqnlf(hc_sg(Rn64Boa#Jg1YYG*$iB6kN*C9x",
    "4(qkdQsyUaj}+rNm9MYiz?pThS!Tqwh1gd-WFiscVxMObe-e1nomtchtZv)k9Mbnopv9Qk81qLN7kyL0KK)XnW7FCYXVlOT^T#3j",
    "Nn?T(kp@+jWaJE+pZZ3bSm)36$(0vtxnNJ4wLYAA4EA*x_1MZ(j2Qrr^rwmJY%%6Et<?Y5JQ}s%NcTN0;hr60VekJvnW;Ni;Jm-n",
    "+;Mqc>b{V-(j)KH>t1{OM7$rrFBRhrtj}T{20CTIzO4AY5a-{3Jz>t7d)$I?J1aT1gplR;bn31}{=2N4ZZ+i7ukR?hTYMS&^Lf4d",
    ")`J??&8$@Agw+*03Z3ScW8Mwi6;PeyXD%i~q_a?e#_wLlaY9^T1>e^RI0~HmJ=8Z6!+iY(bpd%$3#mI0OG{AO8}&;8Ejv=cjUqL>",
    "<V^$9xi!FY@%g-V_1KI_Memg^bz}s6Hs3DuQ?E-B^Mz8M9~AmWcA;Je9X<v79ESaj!FgZLCKk0(#*|F0XUfR>mYG(pH&fXUd9<>$",
    "f-6l6;#VF?bZ;^STCV+`Yc=!@ADpn%IF#lE45U1TpS&zmtc&K0_BqhUUxVO(@w^OsT828(M)Wl<=8^+3y(hGST27Ww|Mg|G9QpG7",
    "VQF0Y&|<dfW}LUCUxxj$l$iGOUFO^2ylq3@&gk{S-cI*Z-?$TR2=Qd14-YKh>wx{GVUX=#u>Obu)MN2D1ocU6EZKoaZs=>FNlz8h",
    "%T=-T%2(N3_hMo<$LN_8dW+|JY*>doJZj(Ed5Sd;KQ4bI>m_{;c{ct7*Ow#ONu1FF`D+8;C*iv%VLw*tFb4#@XlxXDQSZ>bo<z-i",
    "QfP2`6fJC3(2#NQT&$;z*C*?}-4?ASa8thWTth`u_>q!+vCDIxkD8q12e+9c$}@iOxjz01^}24*e=}%t68qX4gL!8aaP^p{JDNti",
    "d(C8cK8)mX=@j#Wo&ruraqr#3uwOl;b^K<Kx%6k#>ZaIL_1gkNROjDJn;6c@{q$k;MSmps^^Y7v9SJhk02$Z}dc+{U>w~!c3H0~7",
    "VJBY#&b$nLs|O?K$VwZHI-;hd7feV~rTp7N65Yy=&8^>^&hO-&NHMY9=Ep|-nl`cE<q$vlpjPbfM~n09@DUdL-VHx=1oXa$bH9ta",
    "_Z;AX`=h@wLPxrj;ba(XqMK*(XrUvIK7SyM+n8+Nmwr+1SiC;D>wz_a-KA^ejF;MsqsV5sPOoJ7sXyO2uP_%ACe9Jyqi)cAFle3x",
    "I+udZKZ3u<Ea=lEl6_tXHP4QuaeNxZED5HPv60-wgcSbi`(oXWCqi58gA&aPS4J)x-(^(Ao-#aMbg`&M9D4_k6zapP2>5oPPUyke",
    "4cNnGJe!04-HZPI`6ARswd8FMrPi@ZQZ850uskiz+7&?We^$a>-ki=}QwF#tbEJ0Jy<sg=M&w@D_7?N@xo&G|>p8WbyzI^#E?%rz",
    "tOLVdq5en1u)Y-gn2kEa4&W%BN+9?9>0}vRNFDkTy6};ehK`M=lrn~!Fh7<*GQ7$wZ7%i5>NVZ6n9RZSk@nD($+^9^e;(=wCv&G?",
    "7U8R_IH1Qa(4`jp8ize-p#y=)7Z6|bdr`0ZEQeY*Al@(sQr68lT7Gu|UGEIz8ov%_Kg<huNWF5M%%ieO+G5txlEeYh;>?w@^F@CA",
    "ex#WHinuQU*JlC!8}Qi?e0CRpu0&q8Ae#6m5$E819^4yI7nDx&raZd5tBiIz)ZD!jqxoZ}(!Em$Xmwk&v>hLv3hE!;-?ns5cI3f`",
    "BN2Y|C`PQ;X+=Ckm2Xa@8Ea=jzVqRizBUop2ldG((I5OKl<dGkcV|V?jv_5BJRU=D-4o7@IA6#Q`XHL=URu?<t0A-|)>=`2^i29f",
    "?qKpgz07()al1;aZ$ycB{t_RKq8n>2!J|X?&gZBz1j7FY0$(@VMy&@Eso{w*GFk&^=)P3?>{1?=_+l8l_E{A(v7n$e?x4Qon=2X9",
    "uXcq@l`;8S|BfU6A3G@(>)vwT{To96*8{uJhVNa4Jj_FG6Rjog5fl0;iR9g>BiC~V>ik+sO}0={eP*Dz|0=ojR|9xgv)cWPq1<`!",
    "cu?J_H5s?}pR?s$iH-WY=^L4!`i4rx1DA{b7j$sptPPOmkvRWQ$lrX-wM{gluYq{u-E^|w2%vUz6m9-hPoqAO)Ac+J_i=A2YyYT}",
    ">AN=3{_4z7!|wKgVVj?h-SF%{+jm#r%JkF!y)(Bc@!^7m{_o%TcD)E=h3>rp{M9{0#4f=c!PoHn!aW#&C`do1m{d0O2V*n1WO)|r",
    "(#A85>QLwO>54X4zqH%3WvSXJLn?Ol$~F3lC-2~(vPB$4mM?x5<|T#uRWSBX60EC6UN$QN^K?n%T#`c#o)VfgMnfkvGAU$19GCdD",
    "g3td;>VD)3b@iVZ-c4@J5s61XPu3m%42=xxVXyu<cU<hd*ToBcZN={$pm7%7zYRap8~yI*kT2ZCoY%%Ea(=6%mKA|C4)fpr9BK*}",
    "8^I0#Eu3Auq0HSrq|meLvbttUQOWe}&1E~!?n^!B%!&3>=f9JO#`*ldSD3egjCJ7I5}ZFC&-x&q#H<7#h5pe1HMua`-yWJq>ws^#",
    "oLEI^CluUbcM5O+pxiMksLXyr8*801N7}UQ?$T7ZJK&vxAE^D{fy=~vFkI~8<oa+>ZSWZ@Aj?}|%d*jz{26mVeSzm66-BKHQmT8~",
    "NLOnMY0oVcJ-RcQW6~7-*jqMc_qHO}V;PcWW`cal3xUaZ_kC0T%=-tk{p6u{aLDn#`A{!Dx5GwR!Fv<z+eXmnS=4iM#l#yy=lA7g",
    "KZ^TCUI?N7(ErQ0`XND;$c>nt#m?KE<GnIE)0GgRXgr&c*z~|c-Nn~WSa&zQl;o#=b*D~P<jeQnp+0g5I7R{Az7+4@DJO0<)_orJ",
    "E6l;Ps`AMi6ih2#i=&18qv#B(3tVU=%fDLgPS#|&TD74KWp9>WSb3NAE<=jCDDIogf0-|f_Vt|woVi=TBUSq3sv0~;o=d;sz7ZSl",
    "-*`NRyt6Q0);ECKGULefOCp`b?Dla_0yp*UQhsY@xcB^RqrG5VTzlzz8HE=dS=%-qj(;g-TcDpgf>hBym-_I}Lf^R+GTtHFg8^L|",
    "CEzw;pO*u#QkV^Ll#mN^sm*$ZreBMqjmMMep}UH?X%4j0qNBXYBZ^xO8Li!=*UL3gX9AKw?wj|JtFzQkz5PzUQ0dE49KxIzzE_L0",
    "HepYDQhj}#aNxe4#oWXV#8<Vc)UJu3ku!4Ok1{AC$;3&*G_353f|*ikvaq)+8WK;aO=Isj|M|>S^WKMwO8wwt<Hh_p%4e_8<0U(E",
    "wi|rcAkI-irl$dKwg<SN=d#HCp_V-J7^+#ANElnAQQv0Kmz6e7;*Djc_9Tb2q0%GSr)|c462l~iWvSEFm?mudDAG^8PAlevLSI9e",
    "mjqqAK>udg5Dk7R@HYXdKV#N|-41;FN5D^hUO?SrW9a5L1vK;-H9h&Zij#7=Jo?8DB*<Nou7YMVST{@dmGqM>P3kQxO76kmihsm8",
    "R$pIIxVOgwyU`-xa;?6)nGF5Gz+mEMXp#RS-(OZt9Y4j;j@OB16jspD&!mXHBY|R;I1s0Gq2Sa3+{+HK6rCe;@kH8@>SxYNA?Eu+",
    "pANkCYJK^F6`#vt%P_M*GfUz3qkwY*4*3S|gNh5Hnz?#9bv=c4HipnWuWHbj)bV7A@KRfND^*t+=qcSW`ZZ=17gzWRug~#gue4%3",
    "DcEOWZUS_*gQxIwNC;4$#N$4QbAYYqLs6fEoO$LYka@P1_PtO>8Ac_gyd1|-3-CzQN;jRebWw!S%H3aC%@2t`$+RfG^X!W9gXh14",
    "XVCcUvoP=MflRl97IL933LE_u{QgDsCFddD7=`;Nit@<%SSa;>zLHL^%b<r10UY`-EbRz(<01hM9WAZp2837hcL(M(Zp%zpVSyj~",
    "<eh#B*5rlz8L&5<Fh>ubOvm|m!ft#N?SnfY#&mCo-AGbV+i`@OFRLgi+(;82RdBR3nWv9*4Cl(!vCV}__j%Jz%Z<_*ZMnMR<8mtf",
    "=uwiG{|fn0iLd_b0-YQ1Oqf^N2A#Tu^A=<7_$u&b4Y*%rMkRG_kD|g$1{zzCP1i>!aI~?6r4dzL?q;rsRVOyO4``-)GD5@J-%I*-",
    "$?60@b<yJg=@E;w0{2Vs0o0D~O$YxmCrRgl8(9zjK8AQwgZZ+jjMRx*(#<OLttM2_7sHY{QpK?Jbex0xHL;cdBd5a~ou;vvwegK5",
    "x<}>jSNMrH@<qE*;kyqCH7RP1^3Cy0$6oFRk1rwKI4sl|%87X<h+J{0WVnFz>9~f9G1q=&c`ip=!g)F@^>X`_PBvTG?!H~V#Ir8)",
    "M%!sa^7QIxKldX=inyRi-#jb>9dhBL4#Zp2vHni%?>y}DINWoR5lxP|O7gU;sV*Uns&Vc?1;G^iI_SJ5h^No>4z3^L;=j$V^-c|L",
    "(tVXTwDHO0{ewS`_LGMy#5%vsH@E1<?_D^1JLIhzds~XnencJkUihQwdE~%soHG&gN5vJSZ^@=Vhg4GfmMD(ftLFI~@r=V^bvox3",
    "8kCRO_UVmf+CMkxR;S;a<R_ji7UNi}IQJyrkq~p>{w3N0IUA1e{{*{{0sotXe)pjg(%rUE+rv>*zfVR7Fsrox@qCWMJRLtM$$>ht",
    "UDs{uXnVvYufJb^ZonbwrQivSA3J&H9t^1u&)|SAw_?2pyx)O+JOvw806aq#@G9s5yHPv0e}lTp0xi|!E{+G62T_$On&ZwWd3Ij1",
    "!~I-Fm;HHFd&8)*arMt7sg6Ax|JSR}XZg|J7_m<$*ei>O+k`)CB%j~E2>Y@c@l_q}N%$4>Wg!vdnHEEhz&!37nM?6wY&7LQBgZ*(",
    "Jo{5F!@MrhdA8-&Dcw?2)A68_Hbc?YfmKpJ{#Wt8yzKA(zJTAr+6G~c4>o%^_VhXCG&iFjdnAawszmaPDyH^tg6Y(G16_N`N<T$Q",
    "IIddD@?M44>r}fu@8%h50@92|F1~C|ul)4$mkRx?n=kq!!S5IQ{E<hv4+{I)fp}mze*TJn_f_Oa0#1gFpw`7<R1={lIq3azK{DN=",
    "4&pd%5z7V_c-<PMtE-V|ue&dL$GU9YFZsh_PxO8v+)thG&YV|-Z$9*I+z8~&EZif97<V}2>^Sr{8t2^|4m_b8ajceF1{iU#nvzy+",
    "N~9sT3OMeOa+V)m<aJyOusFjq+iHFfuUPPK=7TfpgI`;HF33+`GDe(VNbvbz+<!z)$U!Y=T#fb5L9W7aU%~6RH}Cl%;M*DM!fbi%",
    "rU+U$D4ec6i+T3jkmrI>mbDrk%={!vYd=$6!=8jAa&^Mw$d-hORkb00@@}mt&naS`4)()?wcEh+YS`&>*w<Rra}EMOxiXO4!1lNP",
    "m`v8}B3clePYd-5N=pEL-^yY6p@|NrD;n3=Nv(ztt>Z2{l>N5!cK%DUkIMbvy^H^s9|>{01GKgXb6&8I)z|}cmN4JMy#ky}ObIdB",
    ">ExNhkQH{j{$c_x?FymS4wP}+OF=9j63Dp2G_AU_*pB9n+L4ofOFFsbTgf}6?a_X4eRptBF(OV?m_N$)^__RXZZE}teg%HxL)??_",
    "KJKquSV{H|RHOus@ai{lwCB%C`r$q`$F0ud**`P9%w9v6&dHd~-xlj{ziJwFZnku8XlRC?Iz!k$?u97z^+Sa^VJ+S-5$dt9pO0hi",
    "*n~N+LfjLG8Ii88NUHk={e=rg8t`Bh?y<?@9A##{tF_Qu_j{^NKUiKfU{&CX-V?I54t-uqm%7YPzMn7JNvSwTU*^lpFuz6@@SS6z",
    "$wbUsC83XF!Mz`uxOZz|0yW<ZrAd3tG-|Jjp57D2F?&mSPd}}rS!4A~$_Z#6^}bpD^cd?b_v_K$xJH@%;DK{QoZB70pXj^Ktp$5p",
    "0{+j#-+Yh0vmE`DK-fuK8>WK{n$d$@0Bp;=k~n&$yPR{cEM;}33da0(scu2vl(w+>wgq2aH_k(!=Ka6>yMFuoabjImz{w!KalwBy",
    "!j4>k9}L6U??qhq8t^K8Fh7VJQ-wNVZ9*bVdbWgykBO!953)J8*~Yd$sCLx;pt4MAP3b;2sbbvsRT&p)q4ELxtlZDNOsap>MT30v",
    "bX}0AcKD=`*ynhB=S|?g?nWIcEuXvtksm#hObx?Isp$~~<(x>OUa>`-cR~U03Q6%cOjX;P*mCo#k19_|qO^;-?tn||zC=HGQK{(n",
    "qkQ}o>in&sg#kW!iP~2;dmMRaHE;#P1IaNTbBm)?)VwH`)?Luh#lhiJzB`3;d|JRdw`6&n-&X6^J)f$4YGF##L~r_O?pVqr?CdB%",
    "b}~kc+ats}1Nhmh^!1%bg70Ht%ML1tI|{tY3E+aBi6F~YHfk6ijk-!8J+(W9;_NtYYz%8rRe8Hhq?U?0W8Dl^IxIUyf2{fIm|gwP",
    "RQwCBK3c5jD17l`E9{gKdow|{egVxt1D-zu^1L~l7}#ygu~2G!q>A+a4WOOb2HXQ#!ZD9oc}r=sS9vYNGI(uz&Bb8F$ahWZ;+7X=",
    "Crr<n{-u5u=c}8s`M&!`gg$!*{62a`G*6iChTq?Sy7%{aC=Ufu*F+mBfpr|$1bjQ@D-K7PIJd2m?XpKZ8sm%Xg}tKcGE(ElCBIcV",
    "+T5UjeaDOb@UdlLJx9Pl3-XM;SfE>0d~YPqJ6^cA2ldIf5I=v9I>YFAvS)@<dkyX<EDxb$7bNujh9r(@DB|sk40laHqGi&Iq|RTe",
    "^Dk6gh+0^;HuUSP-;(^)RqoVtaz(oVzvP5n>k#A__No|qhS@c)7=5}4YOo=NoWG?|GwxhH^=&%+>Bu74+6>NJ9?m=8&T|-AqU?({",
    "m3E#!Y8hP6D76{PN#87*7~{u2hl%!C<(p3t?%UMi^G3*&39|km&i*{+VIM$zb=3?rY9vo%1T`d=(Gg%*<~=K+Plo6@_j5_S=WwvM",
    "J14AbL9wD{&9^1`(gCvN-HNiA3#55|>N)ZMz<Wck9i`%&2+n>1>;3|IJc_#6Sk%4W4<`oqj=5C1fP)lM6XLJPkva6v2FR7Qh<9zz",
    "cegCjc>3#-8#d&qF6`<I+t$*r^!@%P3jEx+nI!s!Dqr79xCc_`(=|e;mS7)a5r4cJkMtPx8iz2SGCGu;Kc|v)81CQLA4OxOaWr8{",
    "4Cn0=%39JwnA-fju3^^H#(t+&D~2=$O`f-`{I9;_^ZcxPXZ}~9bDpmrYX|LXA=^vvJRh_V0$yceDDj7@02huXk4!@iieg-sTtF{R",
    "H_-VV2F~G?vpQWm)BK03>(tw6ZPKzl<Mp2lmd9MHnEBK9;ePs|@&D_y3-iB!-+zU@bYp$c@NYU_M_gBiIiM-PZT7+a8%z-D-X&zR",
    "S?OiDiFQdNIY(e7uX~O0D(^CSZhn-~5cZB_QS!T{<+pUoA%V4V|8n1Eu!uK{^y#ltxF-<rFW~bB!Q&9v$=`re4NM_7@+ePr7Bw!;",
    "royf~`t$cHdVH#yW41=}x=XoU^Wf0d3vX1k{bEfj{Pb*9zsZB6-~Diw*^fN`b511R*Jsy3rmV0d7hq>LVEyM|CpT$`UkV)KAqjcj",
    "FC_!;=GD(f(2dF@y8c8J=lxO5>q4~NhW)ax1<w>VkNP5@{@G-C#*O&~_4Qkse(sSg6YZo*#EnRNbDCy_uP*u^^!y9tzkfs#i@7?c",
    "F@rn>Ix_56(>~mnao{~vRkLC^cXA2udMJQtvuZ8FY+;?h*mNWFs)Cn~|1$59F|QW-!Fw0~;~pxk&y@P^R}t=Uy@0(80`D=W#jV2q",
    "8@RW}(TMyP_kPv=l0xz`$uxd^CcXJ#5a&IT!`i>scpGcX_Q7AuJ0_n?%8DJLPMU0vPl;Wn`Imb?0>u6xK4<XWDd4Yg&I?xGT-yZX",
    "N2hUK%+kA)3ouWYMeSD-X$NwP54v=;z6JU#tKdBp$#hbcT|Ygqef-WO{YOt)q{j{`Bzcdj{j3`)<_lPZ748iv@ZJC12Kico^B)6k",
    "p26JH0L;VI;+}*s;0gOhkooaq+82>Xj}6e${b!&@gDQFZX8}yj4~%`$tgPC|i6s?>r-Th3e^1WrY3F19WiF;j?BfV^Gw{`gGj?Jx",
    "3qXs@u-EV49=U7f#J^R6d)2hm8Wu<m*(R#LA)$=_7Sr5VE!U7OXX|f-Ix_y7<C+_r(b13{wxIroX--B}METrr$WL5%2Pb3o-Pb4J",
    "fm`7xnz8me_~eDiC)NX>vk>c+*vN4j_lo1*CaVf_^h1+r&Vmq%x)#P+Rk3XKP#rTzpRS9_3Gc9GB%Z3yplx$jM7<YvImnOxhKY5v",
    "eBYcG?qesrke5N;24hb-u$_m2Q+*cx=tuOAzDOZmf{of|m(gJ0m+mr|DRf{Z*S^xs57q^G@6r^uh6Y5ox69I|51y8iv29oUmjf?Y",
    "{p8&lqJ1tE@q}969G}ojTn8N!?lLYye}6RMI#&gG-zy;}?uBbPQ$p*kTH5lxo>mXSJ==d3@ar10nJvBYbgP%D%`HC~>(>P*?6l1+",
    "i~ez2h@bfCPMlLF_K$?QC)l+P;eHjf5C2?%e4!loF-jD;ZWZ%%Q-i2+M?BT1sc7l3RQh#a8P~W-#ny+*yi4oLEx*1Q-mZK#z?Ah=",
    "p-QoV`T5f1a6h=9JGefh4+kaOw`qs{=*0UY@cB7-|04R%x2)(dpnueqOLok5)nY!hIy09B-6*B#iY%^nLIz*`K>>59rO0)nAg!Yz",
    "j#;4Z3?4M=>#E1!UupW6K6}ak?qg&?S37jLS(q;auTSE;1*j8#f;o{t!pL!6hx>tb)V!~h3irYfqJH``a9_=HQrW^RnfJyfg>%`A",
    "vf73n(nY3^il?gdC7<>F!{40soqKB|#rw>$cB|kA!S_OZ9)rILMclp@{m?HF*G&y3U9Uv4qRv$hoK8-^7`mq+g0l`UX6tW8xL5ri",
    "V41l@W-!b*UfA3eb*nhmD*HWD>jwvQXI@gkxe4>m(5Y6@?+f<^9DvLXLB5awKLI<tlHB-HT|3x+*hb?X$)nM@gEA#8lIti6;>Y!~",
    "G1tPPa1W%cvHOwCYSZYVJ(0`89`DuV@4iiyXrCGJK1Si*o+w}btA{>kU@t!bpYt&A&!gb?kCc)f^9!BQR4SCD)5#ejw0ma`*Xbx`",
    "3yZ_OL%)<bUw^N>rSZ+udeg_M;fL%=p({h^U-T&17f=3u-?hev&uIhw3!z)55f}79odFmM_9p!P?R;t-oIrIg6*Or=I4wo(HunBP",
    "uJb@TUoktCxw}4D_vs^5W=>JipPe4{fw$T?u+@>|2ZyW>^H3p<P51TLT`|7;)nLf^%lLjT)US~D^Dm;Gg1Fi8Njf#ZnoJeD!ss$)",
    "Rjwb7<P2Yw1D6}@-m_lnI+LSq<~G}o@JsUJ-Q#5?T{R{@dAC$tH`q5nDA+3@em3Jd@_W)kr*|Qa^~S($pk8-NLS5N<szDw0X0Vo0",
    "dxg-Hq%yAkg?xTpPL@Ma5ab!7);Dr@#}8)RMGv|Qt<SZ-p6RFVeW%VZ;8n`Rdb<!mOMLhZ;7}<Lxa*my6K;p!AFd<kcnx)0<uqNB",
    "LMJ0D=(YO|TwO^iyCb&9aqF?v)}JRE8>yvo9m&&|($tWLsdt>8zE!@6`x4d#o!vtH8SmS`b6}ikA^02sdnJ(rN0CYnIpSE<$lJH2",
    "(S;`!6!U_DK7Lcf)%+gKF8M{vynJ7T^YFr$8YtjqDE>t>=#TzgfBb*uMDE<@7VDEMFZR<4KhcPvHz8xV2brdzKe!9?l4-a<$A!6=",
    "Xw0Wvsib;&JlO)HXf~I^byi2QE1nK;pZ+<c6*pdXU>@!&y{0_CeHHQ}zc|SczAjAcFXW5&=VXa>e*8WTa<CD50)CmpEG4@P>mEYC",
    ">}m=6uo60TUpnRdDW`iDCvw&Y^4LW=Y3>69%=U*S#<kPA(*E>w-YD)?!l!I%)xZ4z2&%+*(%{3nd7!&(p!ovOd?;k`JLI81<9>#E",
    "#8+8r(jCxI_ofumzm-MO$8+iMlTxmsu!tSJ&&ZtH64f<rewx{PB7C!bamcLZcT(reA0a<|t75VKTp`X|CHU@B1&*F>3fqE=Cj!rK",
    "1oKuC3vex$p6t(8P{Xt;y4ok3=6qzMFT<@|cU&2p^?a#&l_K4fJuR!2*_AZX<;@(_u_oqs;d*la_XLKEbrqqG6y(FzTe0>{#6E}}",
    "DFu1TIN&Hg#Qpp`qR8`6DK)Rpq$Oom8qgI?hdV+z!?GfN<hnd2$0o6Ff1$9}yD*~3Q>~L4&YIpFdDP;^FWkBR*(UPW;F}Yv!G16Y",
    "LZk2)_m^{PVXs!9FF7@kTz7|1TS_Vo!@WAkI>PBc1ClsvVKQqR9pK(DBH{n5>*}MLuGjc8Hs0pOHU@i<myI#TI}8~xV6dOzHC#$W",
    "TqQM}w@!J{5Jk&FmOn|<qh?%1#j;#nkD94zzO>`9T1ur^TrSs)d-BSpQsNpdE|!|o{rudI<2g0|;2hXE=llLV-|y%7JfFwB^q0~B",
    "?^^Z#)~`xtS}&EHi+wmbKzw05|6b7W@qK)E2%jStgqC2=M`6GAgyWbY&$<TuN!<A=>(OgyPp`V6q&G}@3cWso-D}l)I#0(tM=z64",
    "Etd^=_saHH?h6Y_NiEvGYJOq>o!+rLTm{cl<@_KQPr(~;1mms1`30Ac#k<5aF9Wn)<XNW%lU1EXQ}4u3J@nl#f1+mHPJw5G&)|wU",
    "SK|ApA<gQ&U|-<dF1=!1QTp1{voXJGcTDd~cn>$@^645lULV%H8hxn8Ihz7J)((D-A%?s^qYe=;v-Ujj81Ja)sUS7UR;$=$r;Q%t",
    "o4Kyhf}BIYKA)WuKT+{{`qk{9)zyhJ`syV2iSO(9dpn1#z-}JG-!*8zn$su5*vG*SLVL>7kwnh!WU4%XYEN}mT5wiG8{g2=E5Agt",
    "<wc>4&1m)RjxuzAv$5pHC$~dwN7mSyhbB!}*>_nHK#wBio<1RiIrVaJ9JKG2`SX7r*z@U#vujcJVPzOOPnXc4H<6zLZHlQ~Qkp!^",
    "PLD3vv0*tG%<Y}w-mFtnpKiCcy+SEi{?3_#+TC^esfjmq0pckp{ysU5*RkRGbtxEc8^${xKI2#D*wiJ$4}$l#&PL|`VzRa=X$dr6",
    "K3En)XS3{VyDFcV)En+yeJQN<{W%4i-ERnu)7*lJ(zVgghdVU^{BIh+pKbj4<>E)U*PVgyJ&*q{f?s+J?`|>PWj~{COPGx+pHomx",
    "S19f3k&)?SF15T~!n(8c%*aPEt`}VqtqF@`Q{o+}jGS$SJ-cg5WFN~S?_-~petr&TC%L@T?reX2;VAsq3D~fkh%a0OKlXbuc{j&V",
    "L%5dQcspL}kkMjj*a|Mn*qXnFGCQgYU5@Xql?S%PpZV++k>rOpwx9PgMUDU2V-A2nkIl1y=L2J$t(ks(!X@|-V{QI{Jc5O|Pd;V@",
    "-B3ae3lO)OSxC3vwUYx}`*%=JWZ<fj+3<+j+5AGjdGn_^njlxO@zG;>R}5+D=x>T61Mr{6@YpmSzrx(hIX(mWF@m*i!oIV^PS(m0",
    "cUF?~D$3n_1^*jnq1&tC=)}jF^phc--MciCSzuE+hrdpjPm?GH76)g<N`q96$RLIE^_Ii{{%8U}UX!2q<$@nF7yIwAC<j8f&>!fp",
    "vs&b(z6D*TZ%RqN6}Tw0lSDg1s1!aTF2v66ZAoCFI*atj{}Szcth%r~e^#_n`?9)dt6VxQe1$T=``5tZu`zt!gwCI@!RdW1L7#4+",
    "f4||L1MW0~ynX$v1#}1+V4CNO=-Mp{iEk&-+~>`#^+PK&yD!<fX?;xV7i;pvHkD~CUnHh%-1L+9w=dkv`_yZ*^Sn(C-@x1+D)PTm",
    "qR_v3oTr6|57q+*23F@BjG)SWsidixKv6Su=+fa5>eJZSWj*PhNe^k9ljoRP*XIV$o7`u$t^c2Lc=8Hu!TOh?0^mpyyuHfs>+|bj",
    "ug(0syA5-JykDA+_irZb^HeE0w`WtgN=?0Q31}oDordlt&>RN#`HIwIRNKAB-^ugMmnOK6-;|u3e=4mfzax6*+ly2I@VYTvC6UjI",
    ";dCK5olo?!8g18No#(?|bs`RX74N~r;KweIP&;C2wz(#{@Z30hIo!xv4{JT24;sAn&B&+gl$1Z)pLhO+cWH=SAnRkwN&@iA)A)Ur",
    "#Mjjl`s)yB&>staAHsRIBmbxjJSbpluAhKEhvm{T=+|6}%cC@CyzO~AfvwrD^jy3i>>QQ|&BX^zDdO0|UEAIhKFc1k{)^G&2jFe0",
    "_;{J#uUpIEX02$~E%)nRIdI<qzl()lf@dXkYySkiy<bFUsuC%7hl2KjJEzoZ*@2~6X2Ze){i$axt#bxrde>#EC%v>W*Z^MBmo#Fy",
    "&pT9luigZwpHm4Rbpv~Q1nZ0(I{G(o!cgS33BkK=5mRMnCRq`~nH{a5=C4h3PLRd6hbWlOn?qc$o*HNV^jm?(uvxogmM#N=JAzqv",
    ">%ja0_a5UFr15nCu@+|B+vjojcG#^^toan=4+0})<R)^>1a4--ee#-~&cip<j}M~1&B|bVPXv2J`x9LG!=c?@I+O!lPSLl58R6^n",
    "rIwevuNeZo2gN*Z1bc|l`|npAe&vRbIEpcM;j_o_9_-f=GaYpRcjA5pJ=m1*WMo6`#JX5L9a$x2%b8fuwI|~B542{>b+*{ObxXyP",
    "#5(by_j~grtp|+(;@zfu{8+AT%U%96=5{U4v;}*u2{!9T%=2aFsiNMZ>s%`NsuQU^BLp~q7!5w9q?Vp!cHWSQIe9tQ8KkuNZk~<S",
    "IA#RzTAFI#>N*`A?9-P9i2sh=pY{H@vy02)!#K-fmycpUzleL*dhq&sP=D}Y#Mvk4kfy63PGWIavy2R%1k?4W<!pI=p2t!&-dlGt",
    "S^m-FkeZGT-M4u&;``^VHBa1CWxh|{2Q{D1F5vMTwBO6=6Jo6On3vaazrt8O;JiCs&{dlOe(Xglb-ou*Rr?F+!COLhAlc$svZX+O",
    ";{8Zp?rQV0o{y4uta*x7uRR=-CrVEUaK9SEMfJRnjoP2Te*<T_4f8$-+gc1g3N!p*pNO1KN{}ZKO70;QwY{H67vL-Z{$VLw<BMWM",
    "zlM7^ZjJ8V^WWs&>3{>CuqgYjwZ_xc9lx&!G{&DC)6Wt5d7D1iEa*z#wO7!Nqj2D&iYW3lA%A~%5_Lo0wFX*MQ_=SHCsQc;FzjTy",
    ")HA!K&^7en_=YJinR|MbH8%R9?p^B!;fFh_0_nL$@VJ@E->xf%&!b11Be0*f7<($mbHn;Jq2A((WNN)5B#liCnROWrKBT3aq9WEU",
    "6nfg$$Xr#7z4e)=Q{97kX>B*}=$fo0Npq^TngH?8vABlOA7@9MGBR_z)3A>lFy?jG=P!fzC4p}36vSaupht00MD4@r^yFI_YKBH)",
    "qZ~R#ajBlhN|Uqs4*43L^4{j_nVpUM6^>?Iy0-C=y!+q`xqQ1>e*H;~_XRy3lHhzrU_VRY7Z98Cl%p<QmWa$v8oKdmGA&2`X1^+r",
    "$}~A_|JGcltj6Z;&mZS28weWckJsDEqAX4Q?O82lf0hRDzhinMVm{8E=C6AQ-XJZ<cTG4`h-b2>-E}wokfo%CH99JPz(S&gT&ft>",
    "(8R7{wjxsFnOIBS3UOBD#Kp9%A}sXe#9!=HT@MRi6MPjFAkJRO<8|SF9LdGuz?fsax#%`NALZf;z&8rP-+*?Ia~<kxap8yI33O6h",
    "Ob(3y*j63uh)80NT`@Wx7YZtm)n&Cie6sV$c4|KO>_>6q^`j*Lbf?Aq`o{Us5r;Ey{0%N2hts=8y)f#-`R&3xWd`ch6d9;{P7;+z",
    "AdfzG9F_iHrbWBL*wP~j%%Zn5TpzTgHXQbd*It^NJ!x*Ga8y;8tNyOHD1ff1iGP>zI{A8uU!TwopJc<hNAdSg#ObC1&k?}yx8zbo",
    "XAsp~4x!nR$z<58qb=KtS;J5!v&E9+>Uca;{^dS}HE4@1_Pd4Js-{KC?PqVs{U5fo2hR"
)
_DIRECTION_NUMBERS = np.frombuffer(
    zlib.decompress(base64.b85decode("".join(_DIRECTION_B85))),
    dtype="<u4",
).reshape(WIDTH, BITS)


def _scrambled_directions(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return SciPy-compatible LMS directions and digital shift."""
    rng = np.random.default_rng(seed)
    bits = np.arange(BITS, dtype=np.uint32)
    random_bits = rng.integers(0, 2, size=(WIDTH, BITS), dtype=np.uint32)
    shift = np.sum(random_bits * (np.uint32(1) << bits), axis=1, dtype=np.uint32)
    ltm = np.tril(rng.integers(0, 2, size=(WIDTH, BITS, BITS), dtype=np.uint32))
    for i in range(BITS):
        ltm[:, i, i] = 1
    sv = _DIRECTION_NUMBERS.copy()
    for dim in range(WIDTH):
        for direction in range(BITS):
            value = int(sv[dim, direction])
            transformed = 0
            power = 1
            for row in range(BITS - 1, -1, -1):
                bit = 0
                for k in range(BITS):
                    bit += int(ltm[dim, row, BITS - 1 - k] & 1) * ((value >> k) & 1)
                transformed += (bit & 1) * power
                power <<= 1
            sv[dim, direction] = np.uint32(transformed)
    return sv, shift


def sobol_uniform(seed: int, *, scramble: bool = True) -> np.ndarray:
    """Generate exactly 2**12 fixed-dimension Sobol uniforms."""
    if scramble:
        directions, quasi = _scrambled_directions(int(seed))
    else:
        directions = _DIRECTION_NUMBERS
        quasi = np.zeros(WIDTH, dtype=np.uint32)
    out = np.empty((ROWS, WIDTH), dtype=np.float64)
    scale = 1.0 / (1 << BITS)
    out[0] = quasi * scale
    for index in range(1, ROWS):
        direction_index = (index & -index).bit_length() - 1
        quasi ^= directions[:, direction_index]
        out[index] = quasi * scale
    return out


def ndtri_dependency_free(p: np.ndarray) -> np.ndarray:
    """Acklam inverse-normal approximation; no SciPy dependency."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, UNIFORM_LO, UNIFORM_HI)
    out = np.empty_like(p)
    low = p < 0.02425
    high = p > 1.0 - 0.02425
    mid = ~(low | high)
    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p[low]))
        num = (((((-7.784894002430293e-03 * q - 3.223964580411365e-01) * q - 2.400758277161838e00) * q - 2.549732539343734e00) * q + 4.374664141464968e00) * q + 2.938163982698783e00)
        den = ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q + 2.445134137142996e00) * q + 3.754408661907416e00) * q + 1.0)
        out[low] = num / den
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        num = (((((-7.784894002430293e-03 * q - 3.223964580411365e-01) * q - 2.400758277161838e00) * q - 2.549732539343734e00) * q + 4.374664141464968e00) * q + 2.938163982698783e00)
        den = ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q + 2.445134137142996e00) * q + 3.754408661907416e00) * q + 1.0)
        out[high] = -num / den
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        num = (((((( -39.69683028665376 * r + 220.9460984245205) * r - 275.9285104469687) * r + 138.3577518672690) * r - 30.66479806614716) * r + 2.506628277459239) * q)
        den = ((((( -54.47609879822406 * r + 161.5858368580409) * r - 155.6989798598866) * r + 66.80131188771972) * r - 13.28068155288572) * r + 1.0)
        out[mid] = num / den
    return out


def sobol_normal_rows(seed: int) -> np.ndarray:
    """Return 4096 rows in R**256 with norm sqrt(256), ready for +/- pairing."""
    uniforms = np.clip(sobol_uniform(seed), UNIFORM_LO, UNIFORM_HI)
    gaussian = ndtri_dependency_free(uniforms)
    gaussian /= np.linalg.norm(gaussian, axis=1, keepdims=True)
    return (np.sqrt(WIDTH) * gaussian).astype(np.float32)


def antipodal_rows(seed: int) -> np.ndarray:
    rows = sobol_normal_rows(seed)
    return np.concatenate((rows, -rows), axis=0)
