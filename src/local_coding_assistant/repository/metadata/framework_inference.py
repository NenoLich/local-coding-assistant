"""Framework and tool inference from dependencies."""

from local_coding_assistant.repository.models import ProjectInfo


class FrameworkInference:
    """Infers frameworks, tools, and test frameworks from dependencies."""

    @staticmethod
    def infer_python_frameworks(info: ProjectInfo) -> None:
        """Infer Python web frameworks from dependencies.

        Args:
            info: ProjectInfo to update with inferred frameworks.
        """
        if "django" in info.clean_dependencies:
            info.core_frameworks.add("Django")
        if "fastapi" in info.clean_dependencies:
            info.core_frameworks.add("FastAPI")
        if "flask" in info.clean_dependencies:
            info.core_frameworks.add("Flask")
        if "litestar" in info.clean_dependencies:
            info.core_frameworks.add("Litestar")

    @staticmethod
    def infer_python_tooling(info: ProjectInfo) -> None:
        """Infer Python tooling (linters, formatters, test frameworks) from dependencies.

        Args:
            info: ProjectInfo to update with inferred tooling.
        """
        # Test frameworks
        if not info.test_frameworks:
            if "pytest" in info.clean_dependencies:
                info.test_frameworks.add("pytest")
            else:
                info.test_frameworks.add("unittest")

    @staticmethod
    def infer_javascript_frameworks(info: ProjectInfo) -> None:
        """Infer JavaScript/TypeScript frameworks from dependencies.

        Args:
            info: ProjectInfo to update with inferred frameworks.
        """
        if "typescript" in info.clean_dependencies:
            info.core_frameworks.add("TypeScript")
        if "react" in info.clean_dependencies:
            info.core_frameworks.add("React")
        if "vue" in info.clean_dependencies:
            info.core_frameworks.add("Vue")
        if "angular" in info.clean_dependencies:
            info.core_frameworks.add("Angular")
        if "next" in info.clean_dependencies:
            info.core_frameworks.add("Next.js")
        if "svelte" in info.clean_dependencies:
            info.core_frameworks.add("Svelte")

    @staticmethod
    def infer_javascript_tooling(info: ProjectInfo) -> None:
        """Infer JavaScript/TypeScript tooling from dependencies.

        Args:
            info: ProjectInfo to update with inferred tooling.
        """
        FrameworkInference._infer_javascript_linters(info)
        FrameworkInference._infer_javascript_tests(info)
        FrameworkInference._infer_javascript_build_tools(info)

    @staticmethod
    def _infer_javascript_linters(info: ProjectInfo) -> None:
        """Infer JavaScript lint and format dependencies."""
        if "prettier" in info.clean_dependencies:
            info.linter_formatter.add("Prettier")
        if "eslint" in info.clean_dependencies:
            info.linter_formatter.add("ESLint")
        if "biome" in info.clean_dependencies:
            info.linter_formatter.add("Biome")

    @staticmethod
    def _infer_javascript_tests(info: ProjectInfo) -> None:
        """Infer JavaScript test frameworks."""
        if "jest" in info.clean_dependencies:
            info.test_frameworks.add("Jest")
        if "vitest" in info.clean_dependencies:
            info.test_frameworks.add("Vitest")
        if "mocha" in info.clean_dependencies:
            info.test_frameworks.add("Mocha")
        if (
            "playwright" in info.clean_dependencies
            or "@playwright/test" in info.clean_dependencies
        ):
            info.test_frameworks.add("Playwright")

    @staticmethod
    def _infer_javascript_build_tools(info: ProjectInfo) -> None:
        """Infer JavaScript build backend tooling."""
        if "vite" in info.clean_dependencies:
            info.build_backend = "Vite"
        if "webpack" in info.clean_dependencies:
            info.build_backend = "Webpack"
        if "rollup" in info.clean_dependencies:
            info.build_backend = "Rollup"
        if "turbo" in info.clean_dependencies:
            info.build_backend = "Turborepo"

    @staticmethod
    def infer_rust_frameworks(info: ProjectInfo) -> None:
        """Infer Rust frameworks from dependencies.

        Args:
            info: ProjectInfo to update with inferred frameworks.
        """
        FrameworkInference._infer_rust_web_frameworks(info)
        FrameworkInference._infer_rust_async_and_utilities(info)
        FrameworkInference._infer_rust_database_tools(info)
        info.test_frameworks.add("cargo test")

    @staticmethod
    def _infer_rust_web_frameworks(info: ProjectInfo) -> None:
        """Infer Rust web frameworks."""
        if "axum" in info.clean_dependencies:
            info.core_frameworks.add("Axum")
        if "actix-web" in info.clean_dependencies:
            info.core_frameworks.add("Actix-Web")
        if "rocket" in info.clean_dependencies:
            info.core_frameworks.add("Rocket")
        if "tauri" in info.clean_dependencies:
            info.core_frameworks.add("Tauri")

    @staticmethod
    def _infer_rust_async_and_utilities(info: ProjectInfo) -> None:
        """Infer Rust async runtimes and helper libraries."""
        if "tokio" in info.clean_dependencies:
            info.core_frameworks.add("Tokio")
        if "async-std" in info.clean_dependencies:
            info.core_frameworks.add("Async-Std")
        if "serde" in info.clean_dependencies:
            info.core_frameworks.add("Serde")
        if "clap" in info.clean_dependencies:
            info.core_frameworks.add("Clap")

    @staticmethod
    def _infer_rust_database_tools(info: ProjectInfo) -> None:
        """Infer database-related Rust frameworks."""
        if "sqlx" in info.clean_dependencies:
            info.core_frameworks.add("SQLx")
        if "diesel" in info.clean_dependencies:
            info.core_frameworks.add("Diesel")
        if "sea-orm" in info.clean_dependencies:
            info.core_frameworks.add("SeaORM")

    @staticmethod
    def infer_go_frameworks(info: ProjectInfo) -> None:
        """Infer Go frameworks from dependencies.

        Args:
            info: ProjectInfo to update with inferred frameworks.
        """
        deps = set(info.clean_dependencies)
        FrameworkInference._infer_go_web_frameworks(info, deps)
        FrameworkInference._infer_go_database_frameworks(info, deps)
        FrameworkInference._infer_go_testing(info, deps)
        FrameworkInference._infer_go_tooling(info, deps)

    @staticmethod
    def _infer_go_web_frameworks(info: ProjectInfo, deps: set[str]) -> None:
        """Infer recognized Go web frameworks from dependency names."""
        framework_map = {
            "gin-gonic/gin": "Gin",
            "gofiber/fiber": "Fiber",
            "labstack/echo": "Echo",
            "go-chi/chi": "Chi",
            "gorilla/mux": "Gorilla Mux",
        }
        for dep_name, framework_name in framework_map.items():
            if dep_name in deps:
                info.core_frameworks.add(framework_name)

    @staticmethod
    def _infer_go_database_frameworks(info: ProjectInfo, deps: set[str]) -> None:
        """Infer database-oriented Go frameworks."""
        framework_map = {
            "gorm.io/gorm": "GORM",
            "jmoiron/sqlx": "SQLx",
            "entgo.io/ent": "Ent",
        }
        for dep_name, framework_name in framework_map.items():
            if dep_name in deps:
                info.core_frameworks.add(framework_name)

    @staticmethod
    def _infer_go_testing(info: ProjectInfo, deps: set[str]) -> None:
        """Infer Go test framework selection."""
        if "stretchr/testify" in deps:
            info.test_frameworks.add("Testify")
        else:
            info.test_frameworks.add("go test")

    @staticmethod
    def _infer_go_tooling(info: ProjectInfo, deps: set[str]) -> None:
        """Infer standard Go tooling from dependencies."""
        info.linter_formatter.update(["gofmt", "go vet"])
        if "golangci/golangci-lint" in deps:
            info.linter_formatter.add("golangci-lint")

    @staticmethod
    def infer_java_frameworks(info: ProjectInfo) -> None:
        """Infer Java frameworks from dependencies.

        Args:
            info: ProjectInfo to update with inferred frameworks.
        """
        deps = info.clean_dependencies

        if any("spring-boot" in d for d in deps):
            info.core_frameworks.add("Spring Boot")
        if any("quarkus" in d for d in deps):
            info.core_frameworks.add("Quarkus")
        if any("micronaut" in d for d in deps):
            info.core_frameworks.add("Micronaut")
        if "hibernate-core" in deps:
            info.core_frameworks.add("Hibernate")

        # Testing
        if "junit-jupiter" in deps or "junit" in deps:
            info.test_frameworks.add("JUnit")
        elif "testng" in deps:
            info.test_frameworks.add("TestNG")

        if "mockito-core" in deps:
            info.core_frameworks.add("Mockito")
