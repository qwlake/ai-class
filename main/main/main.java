package main;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class main {
	
	static int features = 21;
	static int output = 3;

	public static void main(String[] args) {
		
		dataRead dr = new dataRead(features, output);
		LinkedList<double[]> train = dr.readexcel();	// mat 2126
		ArrayList<double[]> test = new ArrayList<double[]>();
		randomSelect(train, test);	// mat 1701, test 425, features 21
		

		
		double[][] trainX = toMatrix(train);
		double[][] trainY = xySplit(trainX);
		double[][] testX = toMatrix(test);
		double[][] testY = xySplit(testX);
		
		network nw = new network(features, features, output);
		double[][] y = nw.predict(trainX);
		nw.meanSquared(y, trainY);
	}
	
	public static double[][] xySplit(double[][] x) {
		double[][] y = new double[x.length][output];
		double[] lineX, lineY;
		for (int i = 0; i < x.length; i++) {
			lineX = new double[features];
			lineY = new double[output];
			for (int j = 0; j < features; j++)
				lineX[j] = x[i][j];
			for (int j = 0; j < output; j++)
				lineY[j] = x[i][j+features];
			x[i] = lineX.clone();
			y[i] = lineY.clone();
		}
		return y;
	}
	
	public static <T> double[][] toMatrix(List<double[]> list) {
		double[][] mat = new double[list.size()][list.get(0).length];
		for (int i = 0; i < mat.length; i++)
			mat[i] = list.get(i).clone();
		return mat;
	}
	
	public static void randomSelect(LinkedList<double[]> mat, ArrayList<double[]> test) {
		for (int i = 0; i < 425; i++) {
			Collections.shuffle(mat);
			test.add(mat.removeLast());
		}
	}
}