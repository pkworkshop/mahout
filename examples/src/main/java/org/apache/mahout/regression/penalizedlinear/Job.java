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

import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.conversion.InputDriver;

import java.io.IOException;

/**
 * Run the penalized linear regression with cross validation.
 * The input file should be Mahout sequence file with VectorWritable or white-spaced TEXT file.
 * In each line, the first element is response; the rest are predictors.
 */
public class Job extends PenalizedLinearDriver {
  private static final String DIRECTORY_CONTAINING_CONVERTED_INPUT = "data";
  private static final String DIRECTORY_CONTAINING_OUTPUT = "output";
  private static String[] jobArgs;
  private static boolean validArgs;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new Job(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    parseJobArgs(args);
    if(validArgs) {
      PenalizedLinearDriver.main(jobArgs);
    }
    return 0;
  }

  private void parseJobArgs(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFile = builder.withLongName("input")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
            .withDescription("where to get training data (Mahout sequence file of VectorWritable or white-spaced TEXT file); in each line, the first element is response; rest are predictors.")
            .create();

    Option outputFile = builder.withLongName("output")
            .withRequired(true)
            .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
            .withDescription("where to get results")
            .create();

    Option lambda = builder.withLongName("lambda")
            .withArgument(argumentBuilder.withName("lambda").withDefault("").withMinimum(0).withMaximum(1).create())
            .withDescription("an increasing positive sequence of penalty coefficient, " +
                    "with length n >= 0; if lambda is not specified, the sequence is chosen by algorithm.")
            .create();

    Option alpha = builder.withLongName("alpha")
            .withArgument(argumentBuilder.withName("alpha").withDefault("1").withMinimum(1).withMaximum(1).create())
            .withDescription("the elastic-net coefficient with default value 1 (LASSO)")
            .create();

    Option bias = builder.withLongName("bias")
            .withDescription("include a bias term")
            .create();

    Option numOfCV = builder.withLongName("numOfCV")
            .withArgument(argumentBuilder.withName("numOfCV").withDefault("5").withMinimum(0).withMaximum(1).create())
            .withDescription("number of cross validation, the rule of thumb is 5 or 10")
            .create();

    Option convert = builder.withLongName("convert")
            .withDescription("pre-processing step if the input file is not Mahout sequence files of VectorWritable: " +
                    "converting space-delimited TEXT file containing floating point numbers into " +
                    "Mahout sequence files of VectorWritable suitable for input of Map-Reduce job.")
            .create();

    addOption(inputFile);
    addOption(outputFile);
    addOption(bias);
    addOption(lambda);
    addOption(alpha);
    addOption(numOfCV);
    addOption(convert);
    if(super.parseArguments(args) == null) {
      validArgs = false;
      return;
    }


    Path input = new Path(getOption("input"));
    Path output = new Path(getOption("output"), DIRECTORY_CONTAINING_CONVERTED_INPUT);
    if(hasOption("convert")) {
      jobArgs = new String[args.length - 1];
      int index = 0;
      for(int i = 0;i < args.length; ++i) {
        if(args[i].equals("--convert")) {
          continue;
        }
        jobArgs[index++] = args[i];
        if(args[i].equals("--input")) {
          args[i + 1] = output.toString();
          InputDriver.runJob(input, output, "org.apache.mahout.math.RandomAccessSparseVector");
        }
        if(args[i].equals("--output")) {
          args[i + 1] = (new Path(getOption("output"), DIRECTORY_CONTAINING_OUTPUT)).toString();
        }
      }
    }
    else {
      jobArgs = args;
    }
    validArgs = true;
  }
}
