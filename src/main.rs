mod research;
mod web;

use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;

use hrm_text_cuda::{ChatSession, HrmTextModel, PromptCondition};

const DEFAULT_MODEL: &str = "sapientinc/HRM-Text-1B";
const DEFAULT_PORT: u16 = 8765;

#[derive(Clone, Copy)]
enum Interface {
    Web,
    Terminal,
}

struct Options {
    model_source: String,
    dev_ordinal: usize,
    interface: Interface,
    port: u16,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut interface = Interface::Web;
        let mut port = DEFAULT_PORT;
        let mut positional = Vec::new();
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--web" => interface = Interface::Web,
                "--terminal" => interface = Interface::Terminal,
                "--port" => {
                    let value = args.next().ok_or("--port requires a value")?;
                    port = value.parse().map_err(|_| format!("invalid port: {}", value))?;
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ if arg.starts_with('-') => return Err(format!("unknown option: {}", arg)),
                _ => positional.push(arg),
            }
        }

        if positional.len() > 2 {
            return Err("too many positional arguments".to_string());
        }

        let model_source = positional.first().cloned().unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let dev_ordinal = positional.get(1)
            .map(|value| value.parse().map_err(|_| format!("invalid CUDA device: {}", value)))
            .transpose()?
            .unwrap_or(0);

        Ok(Self {
            model_source,
            dev_ordinal,
            interface,
            port,
        })
    }
}

fn print_usage() {
    println!("HRM-Text CUDA");
    println!();
    println!("Usage: hrm-text-cuda [options] [model_dir_or_hf_repo] [cuda_device]");
    println!();
    println!("Options:");
    println!("  --web             Launch the local web chat (default)");
    println!("  --terminal        Use the terminal fallback");
    println!("  --port <port>     Web server port (default {})", DEFAULT_PORT);
    println!("  -h, --help        Show this help");
}

fn show_spinner(message: &str) -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("|/-\\ ")
            .template("{spinner:.cyan} {msg:.cyan}")
            .expect("valid spinner template"),
    );
    spinner.set_message(message.to_string());
    spinner.enable_steady_tick(std::time::Duration::from_millis(90));
    spinner
}

fn load_model(options: &Options) -> HrmTextModel {
    let spinner = show_spinner("Loading model...");
    let model = if Path::new(&options.model_source).is_dir() {
        HrmTextModel::from_dir(&options.model_source, options.dev_ordinal)
    } else {
        HrmTextModel::from_hf(&options.model_source, "./hf_cache", options.dev_ordinal)
    };
    spinner.finish_and_clear();

    match model {
        Ok(model) => model,
        Err(error) => {
            eprintln!("{} {}", "Failed to load model:".bright_red().bold(), error);
            std::process::exit(1);
        }
    }
}

fn main() {
    let options = Options::parse().unwrap_or_else(|error| {
        eprintln!("Error: {}", error);
        print_usage();
        std::process::exit(2);
    });

    println!();
    println!("{}", "HRM-Text CUDA".bright_white().bold());
    println!("{}", "Local inference for the HRM-Text base checkpoint".dimmed());
    println!();

    let model = load_model(&options);

    match options.interface {
        Interface::Web => {
            if let Err(error) = web::serve(model, &options.model_source, options.port) {
                eprintln!("{} {}", "Web server error:".bright_red().bold(), error);
                std::process::exit(1);
            }
        }
        Interface::Terminal => run_terminal(model),
    }
}

fn print_terminal_help() {
    println!("{}", "Commands".bold());
    println!("  /help                    Show commands");
    println!("  /clear                   Clear visible conversation history");
    println!("  /context on|off          Include structured history (experimental)");
    println!("  /style reasoning|direct  Change the checkpoint condition");
    println!("  /system <message>        Set or clear an optional prompt prefix");
    println!("  /temp <value>            Temperature; 0 restores greedy decoding");
    println!("  /topk <value>            Top-k; 0 disables it");
    println!("  /topp <value>            Top-p from 0 to 1");
    println!("  /rep <value>             Repetition penalty");
    println!("  /tokens <value>          Maximum generated tokens");
    println!("  /quit                    Exit");
    println!();
}

fn run_terminal(mut model: HrmTextModel) {
    println!("{}", "Terminal mode".bright_cyan().bold());
    println!(
        "{}",
        "This is a pre-alignment base model. Independent prompts are the reliable default.".dimmed()
    );
    println!("Type /help for commands.\n");

    let mut session = ChatSession::new();

    loop {
        print!("{}", "you > ".bright_cyan().bold());
        io::stdout().flush().expect("stdout flush failed");

        let mut input = String::new();
        if io::stdin().read_line(&mut input).unwrap_or(0) == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("/quit") || input.eq_ignore_ascii_case("/exit") {
            break;
        }
        if input.eq_ignore_ascii_case("/help") {
            print_terminal_help();
            continue;
        }
        if input.eq_ignore_ascii_case("/clear") {
            session.clear();
            println!("History cleared.\n");
            continue;
        }
        if let Some(value) = input.strip_prefix("/context ") {
            match value.trim().to_ascii_lowercase().as_str() {
                "on" => session.use_history = true,
                "off" => session.use_history = false,
                _ => {
                    println!("Use /context on or /context off.\n");
                    continue;
                }
            }
            println!("Context mode: {}\n", if session.use_history { "on" } else { "off" });
            continue;
        }
        if let Some(value) = input.strip_prefix("/style ") {
            match value.trim().to_ascii_lowercase().as_str() {
                "reasoning" => session.condition = PromptCondition::Reasoning,
                "direct" => session.condition = PromptCondition::Direct,
                _ => {
                    println!("Use /style reasoning or /style direct.\n");
                    continue;
                }
            }
            println!("Response style updated.\n");
            continue;
        }
        if input.eq_ignore_ascii_case("/system") {
            session.system_prompt = None;
            println!("Prompt prefix cleared.\n");
            continue;
        }
        if let Some(value) = input.strip_prefix("/system ") {
            session.set_system_prompt(value.trim());
            println!("Prompt prefix updated.\n");
            continue;
        }
        if let Some(value) = input.strip_prefix("/temp ") {
            if let Ok(value) = value.trim().parse::<f32>() {
                if value.is_finite() {
                    session.sampler.temperature = value.clamp(0.0, 2.0);
                    println!("Temperature: {:.2}\n", session.sampler.temperature);
                    continue;
                }
            }
            println!("Temperature must be a number from 0 to 2.\n");
            continue;
        }
        if let Some(value) = input.strip_prefix("/topk ") {
            if let Ok(value) = value.trim().parse::<usize>() {
                session.sampler.top_k = value;
                println!("Top-k: {}\n", session.sampler.top_k);
            } else {
                println!("Top-k must be a non-negative integer.\n");
            }
            continue;
        }
        if let Some(value) = input.strip_prefix("/topp ") {
            if let Ok(value) = value.trim().parse::<f32>() {
                if value.is_finite() {
                    session.sampler.top_p = value.clamp(0.0, 1.0);
                    println!("Top-p: {:.2}\n", session.sampler.top_p);
                    continue;
                }
            }
            println!("Top-p must be a number from 0 to 1.\n");
            continue;
        }
        if let Some(value) = input.strip_prefix("/rep ") {
            if let Ok(value) = value.trim().parse::<f32>() {
                if value.is_finite() {
                    session.sampler.repetition_penalty = value.clamp(0.1, 2.0);
                    println!("Repetition penalty: {:.2}\n", session.sampler.repetition_penalty);
                    continue;
                }
            }
            println!("Repetition penalty must be a number from 0.1 to 2.\n");
            continue;
        }
        if let Some(value) = input.strip_prefix("/tokens ") {
            if let Ok(value) = value.trim().parse::<usize>() {
                session.max_tokens = value.clamp(1, 1024);
                println!("Max tokens: {}\n", session.max_tokens);
            } else {
                println!("Max tokens must be a positive integer.\n");
            }
            continue;
        }

        session.add_user_message(input);
        let prompt = session.build_prompt();
        let mut reply = String::new();
        let start = Instant::now();

        print!("{}", "hrm > ".bright_magenta().bold());
        io::stdout().flush().expect("stdout flush failed");

        let result = model.generate_streaming(&prompt, session.max_tokens, &session.sampler, |text| {
            print!("{}", text);
            io::stdout().flush().expect("stdout flush failed");
            reply.push_str(text);
        });

        match result {
            Ok(_) => {
                session.add_assistant_message(reply);
                println!("\n{}\n", format!("Completed in {:.2}s", start.elapsed().as_secs_f64()).dimmed());
            }
            Err(error) => {
                session.messages.pop();
                eprintln!("\n{} {}\n", "Generation failed:".bright_red().bold(), error);
            }
        }
    }

    println!("\nGoodbye.");
}
