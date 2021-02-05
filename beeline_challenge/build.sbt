name := "beeline_challenge"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.5.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.5.1"
libraryDependencies += "com.github.scala-incubator.io" %% "scala-io-core" % "0.4.3"
libraryDependencies += "com.github.scala-incubator.io" %% "scala-io-file" % "0.4.3"

// fix avoid conflict
libraryDependencies += "org.scala-lang" % "scala-compiler" % "2.10.4"
libraryDependencies += "org.apache.commons" % "commons-lang3" % "3.3.2"
libraryDependencies += "org.slf4j" % "slf4j-api" % "1.7.10"