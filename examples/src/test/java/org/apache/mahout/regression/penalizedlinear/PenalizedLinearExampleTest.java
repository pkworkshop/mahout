/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.regression.penalizedlinear;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.regression.feature.extractor.FeatureExtractUtility;
import org.junit.Test;

import java.io.IOException;

public class PenalizedLinearExampleTest extends MahoutTestCase {

  private static final String[] RAW = {"A B C", "1.0 2.0 3.0", "3.0 4.0 2.0", "2.0 1.0 -1.0", "1.5 2.6 0.5", "1.1 2.4 -0.5"};
  private static final double[][] RAW_DATA = {
          {1.0, 2.0, 6.0},
          {3.0, 4.0, 8.0},
          {2.0, 1.0, -1.0},
          {1.5, 2.6, 1.3},
          {1.1, 2.4, -1.2}
  };
  private static final String INDEPENDENT = "B";
  private static final String INTERACTION = "B:C";
  private static final String DEPENDENT = "A";
  private static final String SEP = FeatureExtractUtility.ExtensionToSeparator("txt");

  @Test
  public void testCrossValidation() throws Exception {
    String[] args = {
            "--input", getTestTempFilePath("testdata/file11.txt").toString(),
            "--output", getTestTempFilePath("testdata/file12").toString(),
            "--dependent", DEPENDENT,
            "--independent", INDEPENDENT,
            "--interaction", INTERACTION,
            "--alpha", "0.3",
            "--numOfCV", "2",
            "--bias"};
    Configuration conf = new Configuration();
    try {
      FileSystem fs = FileSystem.get(conf);
      FSDataOutputStream dos = fs.create(getTestTempFilePath("testdata/file11.txt"), true);
      for(int i = 0;i < RAW.length; ++i) {
        dos.writeBytes(RAW[i] + "\n");
      }
      dos.close();
      LinearCrossValidation.main(args);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Test
  public void testRegularizedPath() throws Exception {
    String[] args = {
            "--input", getTestTempFilePath("testdata/file12.txt").toString(),
            "--output", getTestTempFilePath("testdata/file22").toString(),
            "--dependent", DEPENDENT,
            "--independent", INDEPENDENT,
            "--interaction", INTERACTION,
            "--alpha", "0.3",
            "--bias"};
    Configuration conf = new Configuration();
    try {
      FileSystem fs = FileSystem.get(conf);
      FSDataOutputStream dos = fs.create(getTestTempFilePath("testdata/file12.txt"), true);
      for(int i = 0;i < RAW.length; ++i) {
        dos.writeBytes(RAW[i] + "\n");
      }
      dos.close();
      LinearRegularizePath.main(args);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
