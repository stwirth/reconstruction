
#include <Eigen/StdVector>

#ifdef _MSC_VER
#include <unordered_set>
#else
#include <tr1/unordered_set>
#endif

#include <iostream>
#include <stdint.h>

#include "g2o/config.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/slam3d/types_slam3d.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

#if defined G2O_HAVE_CHOLMOD
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#elif defined G2O_HAVE_CSPARSE
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#endif

#include <opencv2/features2d/features2d.hpp>

using namespace Eigen;
using namespace std;


struct View
{
  std::vector<cv::KeyPoint> key_points_left;
  std::vector<cv::KeyPoint> key_points_right;
  std::vector<cv::Point3d> points3d;
  std::string filename;
};

View loadView(const std::string& filename)
{
  View view;
  view.filename = filename;
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  cv::read(fs["key_points_left"], view.key_points_left);
  cv::read(fs["key_points_right"], view.key_points_right);
  fs["points3d"] >> view.points3d;
  return view;
}


int main(int argc, const char* argv[])
{
  if (argc<3)
  {
    cout << endl;
    cout << "Please type: " << endl;
    cout << "sba features_file features_file [features_file ...]" << endl;
    cout << endl;
    exit(0);
  }

  std::vector<View> views;
  for (int i = 1; i < argc; ++i)
  {
    views.push_back(loadView(argv[i]));
  }
  std::cout << "loaded " << views.size() << " views." << std::endl;

  bool DENSE = false;
  cout << "DENSE: "<<  DENSE << endl;

  g2o::SparseOptimizer optimizer;

  optimizer.setVerbose(true);
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

  if (DENSE)
  {
        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
		cerr << "Using DENSE" << endl;
  }
  else
  {
#ifdef G2O_HAVE_CHOLMOD
	cerr << "Using CHOLMOD" << endl;
    linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
#elif defined G2O_HAVE_CSPARSE
    linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
	cerr << "Using CSPARSE" << endl;
#else
#error neither CSparse nor Cholmod are available
#endif
  }

  g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

  g2o::OptimizationAlgorithmLevenberg* optimization_algorithm = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  optimizer.setAlgorithm(optimization_algorithm);

  Vector2d focal_length(779.852,779.852); // pixels
  Vector2d principal_point(522.971,779.852); // 640x480 image
  double baseline = 0.11866;      // baseline

  g2o::ParameterCamera* cam_params = new g2o::ParameterCamera();
  cam_params->setId(0);
  cam_params->setKcam(focal_length[0], focal_length[1], principal_point[0], principal_point[1]);
  optimizer.addParameter(cam_params);

  int vertex_id = 1;
  std::map<int, int> vertex_id_to_point_id;
  for (size_t i = 0; i < views.size(); ++i)
  {
    std::cout << "processing view " << i << std::endl;
    const std::vector<cv::KeyPoint>& key_points_left = views[i].key_points_left;
    const std::vector<cv::KeyPoint>& key_points_right = views[i].key_points_right;
    const std::vector<cv::Point3d>& points3d = views[i].points3d;
    // add camera vertex
    Vector3d trans(0,0,0);
    Eigen:: Quaterniond q;
    q.setIdentity();
    Eigen::Isometry3d cam_pose;
    cam_pose = q;
    cam_pose.translation() = trans;

    g2o::VertexSE3* cam_vertex = new g2o::VertexSE3();
    cam_vertex->setId(vertex_id++);
    cam_vertex->setEstimate(cam_pose);
    if (i == 0) cam_vertex->setFixed(true); // fix the first cam pose

    optimizer.addVertex(cam_vertex);

    // TODO add edges between cam poses
    for (size_t j = 0; j < points3d.size(); ++j)
    {
      std::cout << "processing point " << j << " of view " << i << std::endl;
      g2o::VertexPointXYZ* point_vertex = new g2o::VertexPointXYZ();
      int point_vertex_id = vertex_id++;
      point_vertex->setId(point_vertex_id);
      //point_vertex->setMarginalized(true);
      Vector3d estimate(points3d[j].x, points3d[j].y, points3d[j].z);
      //Vector3d estimate(0, 0, 1); // some wrong values
      point_vertex->setEstimate(estimate);
      optimizer.addVertex(point_vertex);
      vertex_id_to_point_id[point_vertex_id] = j;

      // add edge to camera
      g2o::EdgeSE3PointXYZDisparity* e = new g2o::EdgeSE3PointXYZDisparity();
      // TODO dynamic_cast necessary?
      //e->vertices()[0] = cam_vertex;
      //e->vertices()[1] = point_vertex;
      e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(cam_vertex);
      e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(point_vertex);
      Vector3d z; // measurement
      z[0] = key_points_left[j].pt.x;
      z[1] = key_points_left[j].pt.y;
      double disparity = key_points_left[j].pt.x - key_points_right[j].pt.x;
      z[2] = disparity / (focal_length[0] * baseline); // normalized disparity
      e->setMeasurement(z);
      //e->inverseMeasurement() = -z;
      // variance of the score vector?
      e->information() = Matrix3d::Identity();

      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
      e->setRobustKernel(rk);
      e->setParameterId(0,0);

      optimizer.addEdge(e);
    }
  }


  optimizer.initializeOptimization();
  optimizer.setVerbose(true);

  // first step: structure only optimization, keep camera poses fixed
  /*
  cout << "Performing structure-only BA:"   << endl;
  g2o::StructureOnlySolver<3> structure_only_ba;
  g2o::OptimizableGraph::VertexContainer points;
  for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it) {
    g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
    if (v->dimension() == 3)
      points.push_back(v);
  }

  structure_only_ba.calc(points, 10);

  cout << endl;
  */
  bool saved = optimizer.save("graph.g2o");
  if (!saved) std::cerr << "ERROR during save!" << std::endl;
  cout << "Performing full BA:" << endl;
  optimizer.optimize(10);

  cout << endl;


  /*

  for (tr1::unordered_map<int,int>::iterator it=pointid_2_trueid.begin();
       it!=pointid_2_trueid.end(); ++it)
  {

    g2o::HyperGraph::VertexIDMap::iterator v_it
        = optimizer.vertices().find(it->first);

    if (v_it==optimizer.vertices().end())
    {
      cerr << "Vertex " << it->first << " not in graph!" << endl;
      exit(-1);
    }

    g2o::VertexSBAPointXYZ * v_p
        = dynamic_cast< g2o::VertexSBAPointXYZ * > (v_it->second);

    if (v_p==0)
    {
      cerr << "Vertex " << it->first << "is not a PointXYZ!" << endl;
      exit(-1);
    }

    Vector3d diff = v_p->estimate()-true_points[it->second];

    if (inliers.find(it->first)==inliers.end())
      continue;

    sum_diff2 += diff.dot(diff);

    ++point_num;
  }

  */

  return EXIT_SUCCESS;
}
