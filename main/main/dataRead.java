package main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;

import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;

public class dataRead {
	int features;
	int output;
	public dataRead(int features, int output) {
		this.features = features;
		this.output = output;
	}

	public LinkedList<double[]> readexcel() {
		String path = "./data.xls"; // 읽을 파일 경로
		LinkedList<double[]> mat = new LinkedList<double[]>(); // 데이터를 담을 list

		try {
			File file = new File(path);
			FileInputStream inputStream = new FileInputStream(file);
			HSSFWorkbook hworkbook = new HSSFWorkbook(inputStream); // 2007 이전 버전(xls파일)

			HSSFSheet curSheet; // 현재 sheet
			HSSFRow curRow; // 현재 row

			curSheet = hworkbook.getSheetAt(1);
			int row = curSheet.getPhysicalNumberOfRows() - output;
			// System.out.println(row); // 현재 sheet의 row 갯수 확인
			
			for (int i = 2; i < row; i++) {	// row = 2128
				
				// 0~20(21개) : features
				// 21~23(3개) : answer
				double[] baby = new double[features+output];
				curRow = curSheet.getRow(i);
				for (int j = 0; j < features; j++) {
					baby[j] = Double.parseDouble(String.valueOf(curRow.getCell(j+10).getNumericCellValue()));
				}
				for (int j = features; j < features+output; j++) {
					baby[j] = 0.0;
				}
				if (Double.parseDouble(String.valueOf(curRow.getCell(45).getNumericCellValue())) == 1) {
					baby[features] = 1.0;
				} else if (Double.parseDouble(String.valueOf(curRow.getCell(45).getNumericCellValue())) == 2) {
					baby[features+1] = 1.0;
				} else {
					baby[features+2] = 1.0;
				}
				
//				for (double d : baby) {
//					System.out.print(d + "\t");
//				}
//				System.out.println();
				
				mat.add(baby);
			}

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return mat;
	}
}
