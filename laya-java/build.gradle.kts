plugins {
    `java-library`
}

// JDK 17: the oldest LTS still receiving updates, and the first with records and sealed
// interfaces, which the answer types need -- a ChoiceAnswer is not a ScoreAnswer, and a port that
// models them as one map loses that at the API boundary.
val javaRelease = 17

subprojects {
    apply(plugin = "java-library")

    group = "com.convaiinnovations"
    version = "0.1.0-SNAPSHOT"

    repositories { mavenCentral() }

    extensions.configure<JavaPluginExtension> {
        toolchain { languageVersion.set(JavaLanguageVersion.of(javaRelease)) }
    }

    tasks.withType<JavaCompile>().configureEach {
        options.release.set(javaRelease)
        // -Xlint:all with -Werror: a port's silent narrowing conversion is exactly the class of
        // bug the fixtures cannot see, because it changes a number without changing a shape.
        options.compilerArgs.addAll(listOf("-Xlint:all", "-Werror"))
        options.encoding = "UTF-8"
    }

    tasks.withType<Test>().configureEach {
        useJUnitPlatform()
        testLogging { events("failed", "skipped") }
    }

    dependencies {
        "testImplementation"("org.junit.jupiter:junit-jupiter:5.11.4")
        "testRuntimeOnly"("org.junit.platform:junit-platform-launcher")
    }
}
