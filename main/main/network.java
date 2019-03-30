package main;

import java.util.Random;

public class network {
	int ipSize;
	int hdSize;
	int opSize;
	double[][] wt1;
	double[][] wt2;
	int[] bias;
	double wtConst = 0.01;
	public network(int inputSize, int hiddenSize, int outputSize) {
		this.ipSize = inputSize;
		this.hdSize = hiddenSize;
		this.opSize = outputSize;
		this.wt1 = new double[inputSize][hiddenSize];
		this.wt2 = new double[hiddenSize][outputSize];
		this.bias = new int[hiddenSize];
		
		Random r = new Random();
		for (int i = 0; i < wt1.length; i++)
			for (int j = 0; j < wt1[0].length; j++)
				wt1[i][j] = wtConst * r.nextGaussian();
		r = new Random();
		for (int i = 0; i < wt2.length; i++)
			for (int j = 0; j < wt2[0].length; j++)
				wt2[i][j] = wtConst * r.nextGaussian();
	}
	
	public double[][] predict(double[][] x) {
		double[][] a1 = dot(x, wt1);
		sigmoid(a1);
		double[][] a2 = dot(a1, wt2);
		sigmoid(a2);
		return a2;
	}
	
	public double[][] loss(double[][] x, double[][] t) {
		double[][] y = predict(x);
		return y;
	}
	
	public double accuracy(double[][] x, double[][] t) {
		double[][] y = predict(x);
		int cnt = 0;
		for (int i = 0; i < y.length; i++)
			for (int j = 0; j < y[0].length; j++)
				cnt = (y[i][j] == 1 && t[i][j] == 1)? (cnt+1):cnt;
		return (double)cnt/y.length;
	}
	
	public void gradient(double[][] x, double[][] t) {
		
	}
	
	public double[][] dot(double[][] x, double[][] wt) {
		double[][] a = new double[x.length][wt[0].length];
		for (int i = 0; i < a.length; i++)
			for (int j = 0; j < a[0].length; j++)
				a[i][j] = 1;	// Init the matrix and Add bias 1.0
		for (int i = 0; i < x.length; i++)
			for (int j = 0; j < wt[0].length; j++)
				for (int k = 0; k < x[0].length; k++)
					a[i][j] += x[i][k] * wt[k][j];
		return a;
	}
	
	public double meanSquared(double[][] y, double[][] t) {
		double sum = 0;
		for (int i = 0; i < y.length; i++)
			for (int j = 0; j < y[0].length; j++)
				sum += Math.pow(y[i][j]-t[i][j], 2);
		return 0.5 * sum;
	}
	
	public void sigmoid(double[][] x) {
		for (int i = 0; i < x.length; i++)
			for (int j = 0; j < x[0].length; j++)
				x[i][j] = 1 / (1 + Math.pow(Math.E, -x[i][j]));
	}
	
	public void shape(double[][] arr) {
		System.out.println("[" + arr.length + "," + arr[0].length + "]");
	}
	
	public void print(double[][] arr) {
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr[0].length; j++)
				System.out.print(arr[i][j]);
			System.out.println();
		}
	}
}
