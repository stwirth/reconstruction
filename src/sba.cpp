
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
#include "g2o/types/icp/types_icp.h"
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
}

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

  Vector2d focal_length(550.6352,550.6325); // pixels
  Vector2d principal_point(470,400); // 640x480 image
  double baseline = 0.12;      // 12 cm baseline

  // set up camera params (these are static for all views)
  g2o::VertexSCam::setKcam(focal_length[0],focal_length[1],
                           principal_point[0],principal_point[1],
                           baseline);

  for (size_t i = 0; i < views.size(); ++i)
  {
    const std::vector<cv::KeyPoint>& key_points_left = views[i].key_points_left;
    const std::vector<cv::KeyPoint>& key_points_right = views[i].key_points_right;
    const std::vectro<cv::Point3d>& points3d = views[i].points3d;
    // add camera vertex
    Vector3d trans(0,0,0);
    Eigen:: Quaterniond q;
    q.setIdentity();
    Eigen::Isometry3d cam_pose;
    cam_pose = q;
    cam_pose.translation() = trans;

    g2o::VertexSCam* cam_vertex = new g2o::VertexSCam();
    int vertex_id = 1;
    cam_vertex->setId(vertex_id++);
    cam_vertex->setEstimate(cam_pose);
    cam_vertex->setAll();            // set aux transforms
    cam_vertex->setFixed(true); // fix this vertex

    optimizer.addVertex(cam_vertex);

    // TODO add edges between cam poses

    std::map<int, int> vertex_id_to_point_id;

    for (size_t i = 0; i < points3d.size(); ++i)
    {
      // TODO look if point already in model
      // if not in model, add vertex
      // {
      g2o::VertexSBAPointXYZ* point_vertex = new g2o::VertexSBAPointXYZ();
      int point_vertex_id = vertex_id++;
      point_vertex->setId(point_vertex_id);
      //point_vertex->setMarginalized(true);
      //Vector3d estimate(point3d.x, point3d.y, point3d.z);
      Vector3d estimate(0, 0, 1); // some wrong values
      point_vertex->setEstimate(estimate);
      optimizer.addVertex(point_vertex);
      vertex_id_to_point_id[point_vertex_id] = i;
      //}

      // add edge to camera
      g2o::Edge_XYZ_VSC* e = new g2o::Edge_XYZ_VSC();
      // TODO dynamic_cast necessary?
      e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(point_vertex);
      e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(cam_vertex);
      Vector3d z; // measurement
      z[0] = key_points_left[i].pt.x;
      z[1] = key_points_left[i].pt.y;
      z[2] = key_points_right[i].pt.x;
      e->setMeasurement(z);
      //e->inverseMeasurement() = -z;
      // variance of the score vector?
      e->information() = Matrix3d::Identity();

      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
      e->setRobustKernel(rk);

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
