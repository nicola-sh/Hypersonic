<?php
use Bitrix\Main\Loader;
use Bitrix\Main\Config\Option;

require_once($_SERVER["DOCUMENT_ROOT"]."/bitrix/modules/main/include/prolog_admin.php");

Loader::includeModule('ipol.aliexpress');

$palletId = $_REQUEST['ID'] ?: $_REQUEST['IPOL_ALI_PALLET']['ID'];

if ($palletId) {
	$APPLICATION->SetTitle('Лист передачи #'. $palletId);

	$pallet = \Ipol\Aliexpress\DB\PalletTable::getByPrimary($palletId, [
		'select' => ['*', 'ORDERS'],
	])->fetchObject();
} else {
	$APPLICATION->SetTitle('Создание листа передачи');

	$entityClass = \Ipol\Aliexpress\DB\PalletTable::getObjectClass();
	$pallet = new $entityClass;
	$pallet->fillFromConfig();
}

if (!$pallet) {
    CHTTP::SetStatus("404 Not Found");
    @define("ERROR_404","Y");

    ShowError('Запись не найдена');
    
    exit;
}

$request = \Bitrix\Main\Application::getInstance()->getContext()->getRequest();

$form = new \Ipol\Aliexpress\Admin\Form\Pallet\Edit();
$form->setEntity($pallet);

$result = $form->processRequest($request);

print $form->render($pallet->collectValues(), $result);
