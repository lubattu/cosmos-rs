//! Binary entry point of the reproduction driver.

fn main() -> std::process::ExitCode {
    match cosmos_paper::cli::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(reason) => {
            eprintln!("error: {reason}");
            std::process::ExitCode::FAILURE
        }
    }
}
