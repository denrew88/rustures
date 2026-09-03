use super::*;

#[test]
fn extracts_string_panic_payloads() {
    let borrowed: &(dyn Any + Send) = &"borrowed panic";
    let owned: &(dyn Any + Send) = &String::from("owned panic");
    let opaque: &(dyn Any + Send) = &42_u8;

    assert_eq!(panic_message(borrowed), "borrowed panic");
    assert_eq!(panic_message(owned), "owned panic");
    assert_eq!(panic_message(opaque), "non-string panic payload");
}
