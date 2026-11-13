//! Ray tracing onto a curved Q2 face whose first node is far from the hit
//! (a bug report on GetFEM c5433d32, mwe_raytrace): one 27-node master
//! hexahedron, one slave, zero displacements (the nodes are deformed
//! positions of a twisted bar). GetFEM's Newton starts at the face's first
//! node, converges to a root of the extrapolated face outside it, and
//! loses a penetrating pair; renumbering the master element (same
//! geometry) finds it. The pairs must not depend on the numbering.

use getfem_fem::im_data::ImData;
use getfem_fem::integration::integ_descriptor;
use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::region::RegionSet;
use getfem_gwfl::workspace::Workspace;

const MASTER: [[f64; 3]; 27] = [
  [
    -0.035147140998219376,
    -0.0032143037141210903,
    0.6803581010978857,
  ],
  [
    -0.035163727171192774,
    -0.001781221208898183,
    0.6792075583493452,
  ],
  [
    -0.035155973146328824,
    -0.0008134662771276599,
    0.6778428745882461,
  ],
  [
    -0.03271806926245309,
    -0.0006597967203271007,
    0.6827115638237183,
  ],
  [
    -0.03247779543748119,
    0.0009319778009596706,
    0.6817825136239416,
  ],
  [
    -0.03213639768553028,
    0.002111485538007024,
    0.6806423331414698,
  ],
  [
    -0.029873510612735698,
    0.0014801348399735327,
    0.6856762822792333,
  ],
  [
    -0.02933291015965312,
    0.0030288700250084913,
    0.685166484429626,
  ],
  [
    -0.028834457034033277,
    0.004119721953431667,
    0.6844625140734804,
  ],
  [
    -0.018893808364782157,
    -0.007870494422851648,
    0.6822578898285618,
  ],
  [
    -0.018666808510569698,
    -0.006281146247142034,
    0.6811392973053795,
  ],
  [
    -0.018132097675385907,
    -0.004935977901987715,
    0.6799182705721103,
  ],
  [
    -0.018923275719033184,
    -0.005361657927878755,
    0.6854300181053216,
  ],
  [
    -0.018626057806803348,
    -0.003626179183028825,
    0.6846188063152251,
  ],
  [
    -0.018036681128076052,
    -0.0021517030024598947,
    0.6837134805554745,
  ],
  [
    -0.01857994951275704,
    -0.0032276771933303534,
    0.6893124404930937,
  ],
  [
    -0.018205389566933003,
    -0.001616550164141764,
    0.688985211130029,
  ],
  [
    -0.017665883429333,
    -0.00025665993857569916,
    0.6886532273731643,
  ],
  [
    -0.004313250367613541,
    -0.008490414922049914,
    0.6916753190921512,
  ],
  [
    -0.004370620531130921,
    -0.0069960433354366545,
    0.690616390002924,
  ],
  [
    -0.004105535492846682,
    -0.005456732741359521,
    0.6896948945901207,
  ],
  [
    -0.006497816210846451,
    -0.006897120014931729,
    0.694588108104335,
  ],
  [
    -0.006653590456984819,
    -0.005359127385743116,
    0.6938538348381461,
  ],
  [
    -0.006430673980421575,
    -0.003841975874549021,
    0.6933131601716598,
  ],
  [
    -0.008526878995445086,
    -0.005287165070096231,
    0.6981633951294153,
  ],
  [
    -0.008772622154025814,
    -0.003826572465409246,
    0.6978495364355171,
  ],
  [
    -0.008869473022659918,
    -0.002517598564707752,
    0.6975769538968872,
  ],
];
const SLAVE: [[f64; 3]; 27] = [
  [
    -0.011796007085449214,
    0.0025478414903797077,
    0.6752338674523738,
  ],
  [
    -0.013988521931428084,
    0.0013112483916942839,
    0.6751682598569346,
  ],
  [
    -0.01621210943512879,
    7.938240915678768e-06,
    0.675114618912293,
  ],
  [
    -0.009703271635315001,
    -0.0007554117349449357,
    0.6784110473774981,
  ],
  [
    -0.011363529137379251,
    -0.002165031510173814,
    0.6786069005644721,
  ],
  [
    -0.012982113684286076,
    -0.0035972715138000814,
    0.678804657508379,
  ],
  [
    -0.007256928620261154,
    -0.0037258537328544623,
    0.6815038789005261,
  ],
  [
    -0.00822350534466727,
    -0.00504540778836431,
    0.6818761464313463,
  ],
  [
    -0.00905313782651698,
    -0.0062574329752190655,
    0.6822288831871351,
  ],
  [
    -0.01450893882787041,
    0.005891468143335991,
    0.6836246626434106,
  ],
  [
    -0.016612343520935227,
    0.004221111792195798,
    0.6831106225074439,
  ],
  [
    -0.018825786754463107,
    0.002511407154927094,
    0.6825918622544307,
  ],
  [
    -0.012551138470450966,
    0.002808567520343761,
    0.6872731828083993,
  ],
  [
    -0.014087528029670143,
    0.0009482602153015433,
    0.6870561072461051,
  ],
  [
    -0.01562484128812662,
    -0.0009982975803157178,
    0.686910322878615,
  ],
  [
    -0.010187108012572471,
    0.0003444178739016486,
    0.6910152972492434,
  ],
  [
    -0.010972473401531728,
    -0.0011640280101014276,
    0.6911168174538428,
  ],
  [
    -0.011586283144968675,
    -0.0025736227268315683,
    0.6913130404823142,
  ],
  [
    -0.017750987462948946,
    0.007269027589670861,
    0.6919799514184838,
  ],
  [
    -0.019475856003588275,
    0.005205311804047373,
    0.6906452511600949,
  ],
  [
    -0.02137053743843185,
    0.0027160836758731554,
    0.6895273957646376,
  ],
  [
    -0.015748630032198466,
    0.004198089029383902,
    0.6965669119922595,
  ],
  [
    -0.01667357249677743,
    0.002112355733059259,
    0.6956946657784062,
  ],
  [
    -0.017574554399014432,
    -0.00020170639030879164,
    0.6950775058746882,
  ],
  [
    -0.01361166311098902,
    0.001523054755725197,
    0.7015179365969617,
  ],
  [
    -0.013798502350941134,
    -0.0002602905658754019,
    0.7012530434926031,
  ],
  [
    -0.01384253175625549,
    -0.0019928146473266045,
    0.7010058873807217,
  ],
];

const GT: &str = "GT_PRODUCT(GT_PRODUCT(GT_PK(1,2),GT_PK(1,2)),GT_PK(1,2))";

/// The element of the nodes p (local l = a + 3b + 9k), its region 1 the
/// face whose nodes are those of face 0 of the original numbering.
fn element(p: &[[f64; 3]; 27], face_nodes: &[[f64; 3]]) -> Mesh {
  let m = Mesh::new(3);
  let gt = geotrans_descriptor(GT).unwrap();
  let ids: Vec<usize> = p.iter().map(|x| m.add_point(x).unwrap()).collect();
  m.add_convex(&gt, &ids).unwrap();
  let f = (0..6)
    .find(|&f| {
      gt.structure().ind_points_of_face(f).iter().all(|&l| {
        face_nodes
          .iter()
          .any(|x| x.iter().zip(&p[l]).all(|(a, b)| (a - b).abs() < 1e-14))
      })
    })
    .unwrap();
  let mut r = RegionSet::new();
  r.add_face(0, f);
  m.set_region_set(1, r).unwrap();
  m
}

/// (a, b, k) -> (2 - a, 2 - b, k): the same element, another first node
fn rotated(p: &[[f64; 3]; 27]) -> [[f64; 3]; 27] {
  std::array::from_fn(|l| {
    let (a, b, k) = (l % 3, (l / 3) % 3, l / 9);
    p[(2 - a) + 3 * (2 - b) + 9 * k]
  })
}

/// The gaps of the slave points with a partner, by slave point
fn gaps(master: &[[f64; 3]; 27]) -> Vec<(usize, f64)> {
  let gt = geotrans_descriptor(GT).unwrap();
  let face0: Vec<[f64; 3]> = gt
    .structure()
    .ind_points_of_face(0)
    .iter()
    .map(|&l| MASTER[l])
    .collect();
  let slave_face: Vec<[f64; 3]> = gt
    .structure()
    .ind_points_of_face(0)
    .iter()
    .map(|&l| SLAVE[l])
    .collect();
  let (mm, ms) = (element(master, &face0), element(&SLAVE, &slave_face));
  let mfm = MeshFem::new(&mm, 3)
    .with_classical_finite_element(2)
    .unwrap();
  let mfs = MeshFem::new(&ms, 3)
    .with_classical_finite_element(2)
    .unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("um", &mfm).unwrap();
  ws.add_fem_variable("us", &mfs).unwrap();
  ws.add_raytracing_transformation("rt", 3.39e-3).unwrap();
  ws.add_master_contact_boundary_to_raytracing_transformation("rt", &mm, "um", 1)
    .unwrap();
  ws.add_slave_contact_boundary_to_raytracing_transformation("rt", &ms, "us", 1)
    .unwrap();
  ws.add_macro("normal", "Normalized(Inv(Grad(us)+Id(3))'*Normal)")
    .unwrap();
  ws.add_macro("gap", "(Interpolate(X,rt)+Interpolate(um,rt)-X-us).normal")
    .unwrap();
  let im =
    integ_descriptor("IM_PRODUCT(IM_PRODUCT(IM_GAUSS1D(5),IM_GAUSS1D(5)),IM_GAUSS1D(5))").unwrap();
  let mim = MeshIm::new(&ms).with_integration_method(&im).unwrap();
  let imd = ImData::new(&mim, 1, &[]).unwrap();
  let valid = ws
    .interpolation("Interpolate_filter(rt, 1, 1)", &imd, 1)
    .unwrap();
  let gap = ws
    .interpolation("Interpolate_filter(rt, gap, 1)", &imd, 1)
    .unwrap();
  (0..valid.len())
    .filter(|&i| valid[i] > 0.5)
    .map(|i| (i, gap[i]))
    .collect()
}

#[test]
fn raytrace_independent_of_the_first_face_node() {
  let (g0, g1) = (gaps(&MASTER), gaps(&rotated(&MASTER)));
  println!("original numbering {g0:?}\nrotated numbering {g1:?}");
  // the exact intersections of the slave rays with the master face (the
  // report's 7 x 7 Newton): points 1, 7, 8, gaps 1.011e-3, -1.001e-4, 8.070e-4
  let exact = [(1, 1.011e-3), (7, -1.001e-4), (8, 8.070e-4)];
  for g in [&g0, &g1] {
    assert_eq!(
      g.iter().map(|p| p.0).collect::<Vec<_>>(),
      [1, 7, 8],
      "{g:?}"
    );
    for ((_, a), (_, b)) in g.iter().zip(&exact) {
      assert!((a - b).abs() < 1e-6, "{g:?}");
    }
  }
}
