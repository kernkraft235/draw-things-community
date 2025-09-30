import ArgumentParser
import Diffusion
import Foundation
import ModelOp
import ModelZoo
import NNC
import CryptoKit

extension ModelVersion: ExpressibleByArgument {}

// MARK: - CIVITAI API Structures

private struct CivitaiModelVersion: Decodable {
    struct Model: Decodable {
        let name: String
    }
    let model: Model
    let baseModel: String?
    let triggerWords: [String]?
}

// MARK: - API Key Retrieval (cross-platform + macOS-specific Keychain)

func loadAPIKeyFromDotenv() -> String? {
    let home = FileManager.default.homeDirectoryForCurrentUser
    let dotenvURL = home.appendingPathComponent(".env")
    guard let contents = try? String(contentsOf: dotenvURL) else { return nil }
    for line in contents.split(separator: "\n") {
        let parts = line.split(separator: "=", maxSplits: 1)
        if parts.count == 2 && parts[0].trimmingCharacters(in: .whitespaces) == "CIVITAI_API_KEY" {
            return parts[1].trimmingCharacters(in: .whitespaces)
        }
    }
    return nil
}

#if os(macOS)
// Only compile and run this code on macOS
import Security
func getAPIKeyFromKeychain(service: String = "civitai", account: String = "api-key") -> String? {
    var query: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: service,
        kSecAttrAccount as String: account,
        kSecReturnData as String: true
    ]
    var item: CFTypeRef?
    let status = SecItemCopyMatching(query as CFDictionary, &item)
    guard status == errSecSuccess, let data = item as? Data else { return nil }
    return String(data: data, encoding: .utf8)
}
#endif

func getAPIKey() -> String? {
    // 1. ENV
    if let envKey = ProcessInfo.processInfo.environment["CIVITAI_API_KEY"], !envKey.isEmpty {
        return envKey
    }
    // 2. .env file
    if let dotenvKey = loadAPIKeyFromDotenv(), !dotenvKey.isEmpty {
        return dotenvKey
    }
    #if os(macOS)
    // 3. Keychain (macOS only)
    if let keychainKey = getAPIKeyFromKeychain(), !keychainKey.isEmpty {
        return keychainKey
    }
    #endif
    // Not found
    return nil
}

// MARK: - SHA256 Calculation

func sha256Hash(of fileURL: URL) throws -> String {
    let data = try Data(contentsOf: fileURL)
    let hash = SHA256.hash(data: data)
    return hash.compactMap { String(format: "%02x", $0) }.joined()
}

// MARK: - Civitai API Fetch

enum CivitaiAPIError: Error {
    case missingAPIKey
    case networkError(String)
    case invalidResponse
}

func fetchCivitaiInfo(byHash hash: String, apiKey: String?) async throws -> CivitaiModelVersion? {
    var components = URLComponents(string: "https://civitai.com/api/v1/model-versions/by-hash/\(hash)")!
    var request = URLRequest(url: components.url!)
    if let apiKey = apiKey {
        request.addValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }
    do {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CivitaiAPIError.invalidResponse
        }
        if httpResponse.statusCode == 401 {
            throw CivitaiAPIError.missingAPIKey
        }
        if httpResponse.statusCode != 200 {
            throw CivitaiAPIError.networkError("Received status code \(httpResponse.statusCode)")
        }
        return try JSONDecoder().decode(CivitaiModelVersion.self, from: data)
    } catch {
        throw CivitaiAPIError.networkError(error.localizedDescription)
    }
}

@main
struct Converter: AsyncParsableCommand {
    @Option(
        name: .shortAndLong,
        help: "The LoRA file that is either the safetensors or the PyTorch checkpoint.")
    var file: String
    @Option(name: .shortAndLong, help: "The name of the LoRA.")
    var name: String?
    @Option(help: "The model version for this LoRA.")
    var version: ModelVersion?
    @Option(help: "The model network scale factor for this LoRA.")
    var scaleFactor: Double?
    @Option(name: .shortAndLong, help: "The directory to write the output files to.")
    var outputDirectory: String

    private struct Specification: Codable {
        var name: String
        var file: String
        var version: ModelVersion
        var TIEmbedding: Bool
        var textEmbeddingLength: Int
        var isLoHa: Bool
        var triggerWords: [String]?
        var baseModel: String?
    }

    mutating func run() async throws {
        ModelZoo.externalUrls = [URL(fileURLWithPath: outputDirectory)]
        let fileName = Importer.cleanup(filename: name ?? "unknown") + "_lora_f16.ckpt"
        let scaleFactor = scaleFactor ?? 1.0

        // Step 1: Calculate SHA256 and fetch Civitai info
        let fileURL = URL(fileURLWithPath: file)
        let hash = try sha256Hash(of: fileURL)
        let apiKey = getAPIKey()
        var civitaiInfo: CivitaiModelVersion? = nil

        // Robustness: Informative messaging if API key missing or API call fails
        if apiKey == nil {
            print("Warning: No CIVITAI_API_KEY found in environment, .env, or Keychain (macOS only). API call will proceed unauthenticated (may be rate-limited or restricted).")
        }
        do {
            civitaiInfo = try await fetchCivitaiInfo(byHash: hash, apiKey: apiKey)
        } catch CivitaiAPIError.missingAPIKey {
            print("Error: CIVITAI API key is required but was not found or is invalid. Please set CIVITAI_API_KEY in your environment, .env file, or Keychain (macOS only).")
        } catch CivitaiAPIError.networkError(let message) {
            print("Warning: Could not fetch Civitai metadata: \(message)")
        } catch {
            print("Warning: Failed to fetch Civitai metadata: \(error)")
        }

        // Step 2: Use API data if CLI args not specified
        let resolvedName = name ?? civitaiInfo?.model.name ?? "unknown"
        let resolvedBaseModel = civitaiInfo?.baseModel
        let resolvedTriggerWords = civitaiInfo?.triggerWords

        // Step 3: Import
        let (modelVersion, didImportTIEmbedding, textEmbeddingLength, isLoHa) = try LoRAImporter.import(
            downloadedFile: file, name: resolvedName, filename: fileName, scaleFactor: scaleFactor,
            forceVersion: version
        ) { _ in }

        // Step 4: Build specification
        let specification = Specification(
            name: resolvedName,
            file: fileName,
            version: modelVersion,
            TIEmbedding: didImportTIEmbedding,
            textEmbeddingLength: textEmbeddingLength,
            isLoHa: isLoHa,
            triggerWords: resolvedTriggerWords,
            baseModel: resolvedBaseModel
        )
        let jsonEncoder = JSONEncoder()
        jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
        jsonEncoder.outputFormatting = .prettyPrinted
        let jsonData = try jsonEncoder.encode(specification)
        print(String(decoding: jsonData, as: UTF8.self))
    }
}
