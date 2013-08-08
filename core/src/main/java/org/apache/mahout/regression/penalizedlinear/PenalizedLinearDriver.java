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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Run the penalized linear regression with cross validation.
 * The input file should be Mahout sequence file with VectorWritable.
 * In each line, the first element of VectorWritable is response; the rest are predictors
 */
public class PenalizedLinearDriver extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(PenalizedLinearDriver.class);
  private static final int formatWidth = 8;
  private PenalizedLinearParameter parameter;
  private String input;
  private String output;

  private class PenalizedLinearParameter {
    private int numOfCV;
    private float alpha;
    private String lambda;
    private boolean intercept;

    public int getNumOfCV() {
      return numOfCV;
    }

    public void setNumOfCV(int numOfCV) {
      this.numOfCV = numOfCV;
    }

    public float getAlpha() {
      return alpha;
    }

    public void setAlpha(float alpha) {
      this.alpha = alpha;
    }

    public String getLambda() {
      return lambda;
    }

    public void setLambda(String lambda) {
      this.lambda = lambda;
    }

    public boolean isIntercept() {
      return intercept;
    }

    public void setIntercept(boolean intercept) {
      this.intercept = intercept;
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new PenalizedLinearDriver(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    if (parseArgs(args)) {
      buildRegressionModelMR(parameter, new Path(input), new Path(output));
      PenalizedLinearSolver solver = new PenalizedLinearSolver();
      solver.setAlpha(parameter.alpha);
      solver.setIntercept(parameter.intercept);
      solver.setLambdaString(parameter.lambda);
      solver.initSolver(new Path(output), getConf());
      solver.regularizePath(solver.getLambda());
      printInfo(parameter, solver, "path");

      solver = new PenalizedLinearSolver();
      solver.setAlpha(parameter.alpha);
      solver.setIntercept(parameter.intercept);
      solver.setLambdaString(parameter.lambda);
      solver.initSolver(new Path(output), getConf());
      solver.crossValidate();
      printInfo(parameter, solver, "CV");
    }
    return 0;
  }

  private void printInfo(PenalizedLinearParameter parameter, PenalizedLinearSolver solver, String method) {
    if (method.equals("path")) {
      PenalizedLinearSolver.Coefficients[] coefficients = solver.getCoefficients();
      System.out.println("The training path: lambda    beta0    beta");
      double[] lambdas = solver.getLambda();
      for (int i = 0; i < coefficients.length; ++i) {
        StringBuilder line = new StringBuilder(String.format("%" + Integer.toString(formatWidth) + ".5f", lambdas[i]));
        line.append(" ").append(String.format("%" + Integer.toString(formatWidth) + ".5f", coefficients[i].beta0));
        for (int j = 0; j < coefficients[i].beta.length; ++j) {
          line.append(" ").append(String.format("%" + Integer.toString(formatWidth) + ".5f", coefficients[i].beta[j]));
        }
        System.out.println(line);
      }
    } else {
      PenalizedLinearSolver.Coefficients coefficients = solver.getCoefficients()[0];
      double[] trainError = solver.getTrainError();
      double[] testError = solver.getTestError();
      double[] lambdas = solver.getLambda();

      System.out.println("Training and Test Error: lambda    training    testing");
      for (int i = 0; i < lambdas.length; ++i) {
        StringBuilder line = new StringBuilder(String.format("%" + Integer.toString(formatWidth) + ".5f", lambdas[i]) +
            " " + String.format("%" + Integer.toString(formatWidth) + ".5f", trainError[i]) +
            " " + String.format("%" + Integer.toString(formatWidth) + ".5f", testError[i]));
        System.out.println(line);
      }

      StringBuilder model = new StringBuilder("model: ");
      StringBuilder coefficientString = new StringBuilder();
      if (parameter.intercept) {
        model.append("beta0    beta");
        coefficientString.append(String.format("%" + Integer.toString(formatWidth) + ".5f", coefficients.beta0));
        for (int i = 0; i < coefficients.beta.length; ++i) {
          coefficientString.append(" ").append(String.format("%" + Integer.toString(formatWidth) + ".5f", coefficients.beta[i]));
        }
      } else {
        model.append("beta");
        for (int i = 0; i < coefficients.beta.length; ++i) {
          coefficientString.append(" ").append(String.format("%" + Integer.toString(formatWidth) + ".5f", coefficients.beta[i]));
        }
      }
      System.out.println(model);
      System.out.println(coefficientString);

      System.out.println("Optimal lambda is: " + String.format("%" + Integer.toString(formatWidth) + ".5f", solver.getOptLambda()[0]));
    }
  }

  private void buildRegressionModelMR(PenalizedLinearParameter parameter, Path input, Path output)
      throws IOException, InterruptedException, ClassNotFoundException {

    Job job = prepareJob(
        input,
        output,
        SequenceFileInputFormat.class,
        PenalizedLinearMapper.class,
        Text.class,
        VectorWritable.class,
        PenalizedLinearReducer.class,
        Text.class,
        VectorWritable.class,
        SequenceFileOutputFormat.class
    );
    job.setJobName("Penalized Linear Regression Driver running over input: " + input);
    job.setNumReduceTasks(1);
    job.setJarByClass(PenalizedLinearDriver.class);

    Configuration conf = job.getConfiguration();
    conf.setInt(PenalizedLinearKeySet.NUM_CV, parameter.getNumOfCV());
    conf.setFloat(PenalizedLinearKeySet.ALPHA, parameter.getAlpha());
    conf.set(PenalizedLinearKeySet.LAMBDA, parameter.getLambda());
    conf.setBoolean(PenalizedLinearKeySet.INTERCEPT, parameter.isIntercept());

    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("Penalized Linear Regression Job failed processing " + input);
    }
  }

  private boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help").withDescription("print this list").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFile = builder.withLongName("input")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
        .withDescription("where to get training data (Mahout sequence file of VectorWritable); in each line, the first element is response; rest are predictors.")
        .create();

    Option outputFile = builder.withLongName("output")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("where to get results")
        .create();

    Option lambda = builder.withLongName("lambda")
        .withArgument(argumentBuilder.withName("lambda").withDefault("0").withMinimum(1).create())
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

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFile)
        .withOption(outputFile)
        .withOption(lambda)
        .withOption(alpha)
        .withOption(bias)
        .withOption(numOfCV)
        .create();

    Parser parser = new Parser();
    parser.setHelpOption(help);
    parser.setHelpTrigger("--help");
    parser.setGroup(normalArgs);
    parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 130));
    CommandLine cmdLine = parser.parseAndHelp(args);
    if (cmdLine == null) {
      return false;
    }

    parameter = new PenalizedLinearParameter();
    parameter.setNumOfCV(Integer.parseInt((String) cmdLine.getValue(numOfCV)));
    parameter.setAlpha(Float.parseFloat((String) cmdLine.getValue(alpha)));
    parameter.setIntercept(cmdLine.hasOption(bias));

    if (!processLambda(parameter, cmdLine, lambda) || parameter.alpha < 0.0 || parameter.alpha > 1.0 || parameter.numOfCV < 1 || parameter.numOfCV > 20) {
      log.error("please make sure the lambda sequence is positive and increasing, and 0.0 <= alphaValue <= 1.0 and 1 <= numOfCV <= 20");
      return false;
    }

    input = (String) cmdLine.getValue(inputFile);
    output = (String) cmdLine.getValue(outputFile);

    return true;
  }

  boolean processLambda(PenalizedLinearParameter parameter, CommandLine cmdLine, Option lambda) {
    StringBuilder lambdaSeq = new StringBuilder();
    double previous = Double.NEGATIVE_INFINITY;
    if (cmdLine.hasOption(lambda)) {
      for (Object x : cmdLine.getValues(lambda)) {
        double number = Double.parseDouble(x.toString());
        if (previous >= number || number < 0) {
          return false;
        }
        lambdaSeq.append(x.toString()).append(",");
        previous = number;
      }
      parameter.setLambda(lambdaSeq.substring(0, lambdaSeq.length() - 1));
      return true;
    } else {
      parameter.setLambda("");
      return true;
    }
  }
}
